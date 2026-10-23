-- Prefix-reuse patterns over long documents (GovReport, truncated to ~8k chars = ~2k tokens) and product
-- images (SemBench ECOMM). Each pattern is one query whose later AI calls can read a provider-cached span the
-- first call wrote. Run by run.py twice per pattern: prefix cache on, then off (the cost baseline).
-- Markers: $DOCS / $PICS are the row limits run.py substitutes; the question texts carry no knobs here
-- (the mock run adds '[p=NN]' answers through run.py --mock).

-- P1 projection pair, same function on one document: certain reuse, 1 write + 1 read per row
SELECT id,
       ai_filter(report || ' The report discusses a federal program''s budget.') AS budget,
       ai_filter(report || ' The report recommends an action to an agency.') AS recommends
FROM docs;

-- P2 filter then projection, same function: reads only on the survivors (stochastic)
SELECT id, ai_filter(report || ' The report names a specific dollar amount.') AS amount
FROM docs
WHERE ai_filter(report || ' The report concerns health care or public health.');

-- P3 conjunction of three predicates in one node (within-node expectation, per-row order)
SELECT id FROM docs
WHERE ai_filter(report || ' The report concerns the Department of Defense.')
  AND ai_filter(report || ' The report mentions a cost estimate.')
  AND ai_filter(report || ' The report was requested by Congress.');

-- P4 sibling filters over the same scan (ECOMM q10 shape): cross-operator, adaptive today
WITH a AS (SELECT id FROM docs WHERE ai_filter(report || ' The report concerns education.')),
     b AS (SELECT id FROM docs WHERE ai_filter(report || ' The report concerns transportation.')),
     c AS (SELECT id FROM docs WHERE ai_filter(report || ' The report concerns energy.'))
SELECT count(*) FROM a, b, c WHERE a.id < b.id AND b.id < c.id;

-- P5 pair prompts: a document against several short questions (the factor graph declares the document side)
SELECT d.id, q.q FROM docs d, questions q
WHERE ai_filter(d.report || ' Question: ' || q.q);

-- P6 images, two predicates per image (ECOMM)
SELECT id FROM pics
WHERE ai_filter(ai_image(filepath) || ' The product shown is a piece of footwear.')
  AND ai_filter(ai_image(filepath) || ' The product shown is predominantly black.');

-- P7 images, projection pair
SELECT id,
       ai_filter(ai_image(filepath) || ' The product shown carries visible text or a logo.') AS branded,
       ai_filter(ai_image(filepath) || ' The product shown is photographed on a white background.') AS studio
FROM pics;

-- P8 few-shot head: a constant 1,200-token instruction block before each document, one predicate per row
--    (the head is shared by every row; the document span, read by nothing else, is not written)
SELECT id FROM docs
WHERE ai_filter(repeat('Example: a report that sets a deadline for an agency, names the statute it acts under and gives a dollar figure for the programme is specific; a report that only summarises hearings without dates, statutes or figures is not. ', 32) || 'Judge the report below by the examples above. Is it specific? ' || report);
