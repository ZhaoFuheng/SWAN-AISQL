#include "ai_selectivity_model.hpp"

#include <cmath>
#include <cstdlib>

namespace duckdb {

namespace {

// Hidden widths and training hyper-parameters. Named so they are easy to tune in one place.
constexpr idx_t kHidden1 = 256; // wide first layer: the input is a large [pred_emb, input_emb, cos_sim] vector
constexpr idx_t kHidden2 = 64;
constexpr idx_t kFifoCapacity = 64; // fixed-size FIFO of most-recent (embedding, label) pairs
constexpr idx_t kMinTrainSize = 16; // minimum buffered examples before a training step runs
constexpr double kL2 = 1e-5;
// Adam optimizer (adaptive per-parameter rates -> faster, more stable online convergence than SGD).
constexpr double kAdamLR = 0.01;
constexpr double kAdamBeta1 = 0.9;
constexpr double kAdamBeta2 = 0.999;
constexpr double kAdamEps = 1e-8;

inline double SelReLU(double x) {
	return x > 0.0 ? x : 0.0;
}
inline double SelSigmoid(double x) {
	// Numerically stable logistic.
	if (x >= 0.0) {
		return 1.0 / (1.0 + std::exp(-x));
	}
	const double e = std::exp(x);
	return e / (1.0 + e);
}

// He-ish uniform init: range sqrt(6 / fan_in) keeps activations from saturating.
void InitLayer(std::mt19937 &rng, vector<float> &w, vector<float> &b, idx_t out_dim, idx_t in_dim) {
	const double limit = std::sqrt(6.0 / static_cast<double>(in_dim));
	std::uniform_real_distribution<double> dist(-limit, limit);
	w.resize(out_dim * in_dim);
	for (auto &v : w) {
		v = static_cast<float>(dist(rng));
	}
	b.assign(out_dim, 0.0f);
}

} // namespace

//===--------------------------------------------------------------------===//
// Forward pass (inference)
//===--------------------------------------------------------------------===//
double AISelectivityParams::Forward(const float *embedding, idx_t dim) const {
	if (input_dim == 0 || dim != input_dim) {
		return 0.5;
	}
	vector<double> a1(h1);
	for (idx_t i = 0; i < h1; i++) {
		double acc = b1[i];
		const float *row = &w1[i * input_dim];
		for (idx_t j = 0; j < input_dim; j++) {
			acc += static_cast<double>(row[j]) * static_cast<double>(embedding[j]);
		}
		a1[i] = SelReLU(acc);
	}
	vector<double> a2(h2);
	for (idx_t i = 0; i < h2; i++) {
		double acc = b2[i];
		const float *row = &w2[i * h1];
		for (idx_t j = 0; j < h1; j++) {
			acc += static_cast<double>(row[j]) * a1[j];
		}
		a2[i] = SelReLU(acc);
	}
	double z = b3[0];
	for (idx_t j = 0; j < h2; j++) {
		z += static_cast<double>(w3[j]) * a2[j];
	}
	return SelSigmoid(z);
}

//===--------------------------------------------------------------------===//
// Model lifecycle
//===--------------------------------------------------------------------===//
AISelectivityModel &AISelectivityModel::Global() {
	static AISelectivityModel instance;
	return instance;
}

std::shared_ptr<const AISelectivityParams> AISelectivityModel::GetParams() {
	std::lock_guard<std::mutex> lock(params_mutex);
	return params;
}

std::shared_ptr<const AISelectivityParams> AISelectivityModel::EnsureParams(idx_t dim) {
	std::lock_guard<std::mutex> lock(params_mutex);
	if (params && params->input_dim == dim) {
		return params;
	}
	auto next = std::make_shared<AISelectivityParams>();
	next->input_dim = dim;
	next->h1 = kHidden1;
	next->h2 = kHidden2;
	// Reseed from a fixed (env-overridable) seed so the random init is reproducible across runs;
	// set AI_MLP_SEED to vary it for robustness experiments.
	uint32_t seed = 0x5EED1234u;
	if (const char *s = std::getenv("AI_MLP_SEED")) {
		seed = static_cast<uint32_t>(std::strtoul(s, nullptr, 0));
	}
	init_rng.seed(seed);
	InitLayer(init_rng, next->w1, next->b1, kHidden1, dim);
	InitLayer(init_rng, next->w2, next->b2, kHidden2, kHidden1);
	InitLayer(init_rng, next->w3, next->b3, 1, kHidden2);
	params = next;
	return params;
}

double AISelectivityModel::Predict(const float *embedding, idx_t dim) {
	if (dim == 0 || embedding == nullptr) {
		return 0.5;
	}
	auto p = GetParams();
	if (!p || p->input_dim != dim) {
		p = EnsureParams(dim);
	}
	return p->Forward(embedding, dim);
}

void AISelectivityModel::AddExample(const vector<float> &embedding, bool label) {
	if (embedding.empty()) {
		return;
	}
	std::lock_guard<std::mutex> lock(buffer_mutex);
	buffer.push_back(Sample {embedding, label ? 1.0f : 0.0f});
	while (buffer.size() > kFifoCapacity) {
		buffer.pop_front(); // FIFO eviction: keep the most recent kFifoCapacity examples
	}
	examples_seen++;
}

//===--------------------------------------------------------------------===//
// Training: one SGD step on a random mini-batch (BCE loss)
//===--------------------------------------------------------------------===//
void AISelectivityModel::TrainMiniBatchStep() {
	// Snapshot the full FIFO (under buffer_mutex) so the actual gradient work happens lock-free and
	// cannot deadlock against Predict/AddExample. Full-batch gradient -> low variance, more stable.
	vector<Sample> batch;
	idx_t dim;
	{
		std::lock_guard<std::mutex> lock(buffer_mutex);
		if (buffer.size() < kMinTrainSize) {
			return;
		}
		dim = buffer.front().emb.size();
		batch.reserve(buffer.size());
		for (auto &s : buffer) {
			if (s.emb.size() == dim) {
				batch.push_back(s);
			}
		}
	}
	if (batch.empty() || dim == 0) {
		return;
	}

	// Copy-on-write: gradient-step a fresh copy of the current weights, then swap it in.
	auto base = EnsureParams(dim);
	auto next = std::make_shared<AISelectivityParams>(*base);
	auto &p = *next;
	const idx_t h1 = p.h1, h2 = p.h2;

	// Gradient accumulators.
	vector<double> gw1(p.w1.size(), 0.0), gb1(h1, 0.0);
	vector<double> gw2(p.w2.size(), 0.0), gb2(h2, 0.0);
	vector<double> gw3(h2, 0.0);
	double gb3 = 0.0;
	double loss_sum = 0.0;

	vector<double> a1(h1), a2(h2), d1(h1), d2(h2);
	for (auto &s : batch) {
		const float *x = s.emb.data();
		// Forward with cached activations.
		for (idx_t i = 0; i < h1; i++) {
			double acc = p.b1[i];
			const float *row = &p.w1[i * dim];
			for (idx_t j = 0; j < dim; j++) {
				acc += static_cast<double>(row[j]) * static_cast<double>(x[j]);
			}
			a1[i] = SelReLU(acc);
		}
		for (idx_t i = 0; i < h2; i++) {
			double acc = p.b2[i];
			const float *row = &p.w2[i * h1];
			for (idx_t j = 0; j < h1; j++) {
				acc += static_cast<double>(row[j]) * a1[j];
			}
			a2[i] = SelReLU(acc);
		}
		double z = p.b3[0];
		for (idx_t j = 0; j < h2; j++) {
			z += static_cast<double>(p.w3[j]) * a2[j];
		}
		const double y_hat = SelSigmoid(z);
		const double y = s.label;
		// BCE loss (clamped for the log) and its gradient w.r.t. the logit: dL/dz = y_hat - y.
		const double eps = 1e-7;
		const double yc = y_hat < eps ? eps : (y_hat > 1.0 - eps ? 1.0 - eps : y_hat);
		loss_sum += -(y * std::log(yc) + (1.0 - y) * std::log(1.0 - yc));
		const double dz = y_hat - y;

		// Backprop output layer.
		gb3 += dz;
		for (idx_t j = 0; j < h2; j++) {
			gw3[j] += dz * a2[j];
			// grad into a2, through ReLU'.
			d2[j] = (a2[j] > 0.0) ? dz * static_cast<double>(p.w3[j]) : 0.0;
		}
		// Backprop layer 2.
		for (idx_t i = 0; i < h2; i++) {
			gb2[i] += d2[i];
			double *grow = &gw2[i * h1];
			for (idx_t j = 0; j < h1; j++) {
				grow[j] += d2[i] * a1[j];
			}
		}
		// grad into a1, through ReLU'.
		for (idx_t j = 0; j < h1; j++) {
			double acc = 0.0;
			for (idx_t i = 0; i < h2; i++) {
				acc += d2[i] * static_cast<double>(p.w2[i * h1 + j]);
			}
			d1[j] = (a1[j] > 0.0) ? acc : 0.0;
		}
		// Backprop layer 1.
		for (idx_t i = 0; i < h1; i++) {
			gb1[i] += d1[i];
			double *grow = &gw1[i * dim];
			for (idx_t j = 0; j < dim; j++) {
				grow[j] += d1[i] * static_cast<double>(x[j]);
			}
		}
	}

	const double inv = 1.0 / static_cast<double>(batch.size());

	// Adam update, under params_mutex (which also does the COW swap). The moments persist across
	// steps and are (re)sized to match the current param shape on first use / a dimension change.
	{
		std::lock_guard<std::mutex> lock(params_mutex);
		if (adam.dim != dim || adam.h1 != h1 || adam.h2 != h2) {
			adam.dim = dim;
			adam.h1 = h1;
			adam.h2 = h2;
			adam.t = 0;
			adam.m_w1.assign(p.w1.size(), 0.0);
			adam.v_w1.assign(p.w1.size(), 0.0);
			adam.m_b1.assign(h1, 0.0);
			adam.v_b1.assign(h1, 0.0);
			adam.m_w2.assign(p.w2.size(), 0.0);
			adam.v_w2.assign(p.w2.size(), 0.0);
			adam.m_b2.assign(h2, 0.0);
			adam.v_b2.assign(h2, 0.0);
			adam.m_w3.assign(p.w3.size(), 0.0);
			adam.v_w3.assign(p.w3.size(), 0.0);
			adam.m_b3.assign(1, 0.0);
			adam.v_b3.assign(1, 0.0);
		}
		adam.t++;
		const double bc1 = 1.0 - std::pow(kAdamBeta1, static_cast<double>(adam.t));
		const double bc2 = 1.0 - std::pow(kAdamBeta2, static_cast<double>(adam.t));
		auto adam_step = [&](vector<float> &w, const vector<double> &g, vector<double> &m, vector<double> &v,
		                     bool decay) {
			for (idx_t k = 0; k < w.size(); k++) {
				const double grad = g[k] * inv + (decay ? kL2 * static_cast<double>(w[k]) : 0.0);
				m[k] = kAdamBeta1 * m[k] + (1.0 - kAdamBeta1) * grad;
				v[k] = kAdamBeta2 * v[k] + (1.0 - kAdamBeta2) * grad * grad;
				const double mhat = m[k] / bc1;
				const double vhat = v[k] / bc2;
				w[k] = static_cast<float>(static_cast<double>(w[k]) - kAdamLR * mhat / (std::sqrt(vhat) + kAdamEps));
			}
		};
		adam_step(p.w1, gw1, adam.m_w1, adam.v_w1, true);
		adam_step(p.b1, gb1, adam.m_b1, adam.v_b1, false);
		adam_step(p.w2, gw2, adam.m_w2, adam.v_w2, true);
		adam_step(p.b2, gb2, adam.m_b2, adam.v_b2, false);
		adam_step(p.w3, gw3, adam.m_w3, adam.v_w3, true);
		vector<double> gb3v(1, gb3);
		adam_step(p.b3, gb3v, adam.m_b3, adam.v_b3, false);
		params = next; // COW swap; readers holding the old shared_ptr keep a valid snapshot
	}
	{
		std::lock_guard<std::mutex> lock(buffer_mutex);
		const double loss = loss_sum * inv;
		if (first_loss < 0.0) {
			first_loss = loss;
		}
		last_loss = loss;
		train_steps++;
	}
}

std::thread AISelectivityModel::LaunchOverlappedTraining() {
	{
		std::lock_guard<std::mutex> lock(buffer_mutex);
		if (buffer.size() < kMinTrainSize) {
			return std::thread {}; // nothing to train on yet
		}
	}
	bool expected = false;
	if (!training_in_flight.compare_exchange_strong(expected, true)) {
		return std::thread {}; // another trainer is already running
	}
	return std::thread([this]() {
		TrainMiniBatchStep();
		training_in_flight.store(false);
	});
}

void AISelectivityModel::TrainInlineIfReady() {
	bool expected = false;
	if (!training_in_flight.compare_exchange_strong(expected, true)) {
		return; // another thread is already training
	}
	TrainMiniBatchStep();
	training_in_flight.store(false);
}

AISelectivityStats AISelectivityModel::Stats() {
	AISelectivityStats stats;
	{
		std::lock_guard<std::mutex> lock(buffer_mutex);
		stats.buffered = buffer.size();
		stats.examples_seen = examples_seen;
		stats.train_steps = train_steps;
		stats.first_loss = first_loss;
		stats.last_loss = last_loss;
	}
	auto p = GetParams();
	stats.input_dim = p ? p->input_dim : 0;
	return stats;
}

uint64_t AISelectivityModel::Reset() {
	uint64_t cleared;
	{
		std::lock_guard<std::mutex> lock(buffer_mutex);
		cleared = buffer.size();
		buffer.clear();
		examples_seen = 0;
		train_steps = 0;
		first_loss = -1.0;
		last_loss = -1.0;
	}
	{
		std::lock_guard<std::mutex> lock(params_mutex);
		params.reset();
		adam = AdamState {}; // drop optimizer moments so fresh weights start with a clean Adam state
	}
	return cleared;
}

} // namespace duckdb
