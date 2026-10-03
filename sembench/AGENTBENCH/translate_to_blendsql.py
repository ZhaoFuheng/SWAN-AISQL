#!/usr/bin/env python3
"""Mechanical translation of the hybrid bench's SWAN queries (swan_queries/QN.sql) into BlendSQL.

BlendSQL evaluates an LLM ingredient over ONE column of a table or CTE, so every SELECT that holds an AI
call is split in two: a base CTE that keeps the relational part and renders each call's prompt into a text
column, and the original SELECT rewritten to read from it with `{{LLMMap(...)}}` in place of the call.

    ai_filter('<question>\\n Label: ' || a || ' ... ' || b || <PLOP suffix>)
        -> {{LLMMap('<question>', base.__p0)}} = TRUE          with base.__p0 = concat('Label: ', a, ' ... ', b)
    COALESCE(TRY_CAST(regexp_extract(ai_complete(...), ...) AS INTEGER), 0)
        -> COALESCE({{LLMMap('<question>', base.__p0, return_type='int')}}, 0)
    ai_complete(<prompt>)
        -> {{LLMMap('<question>', base.__p0, return_type='str')}}

The question is the prompt's leading literal up to its last line break; the context column is the rest of
the prompt (concat, NULL-tolerant like PLOP's). PLOP's answer-format suffix is dropped: BlendSQL adds its
own. An AI conjunct in an inner JOIN ... ON is moved to WHERE; a derived table holding an AI call becomes a
CTE. Setup statements (CREATE TABLE/VIEW over the dataset files) are returned separately for the backend.

    python3 translate_to_blendsql.py Q4         # print the translation
    python3 translate_to_blendsql.py            # translate all 30 to blendsql_queries/QN.sql
"""
import os
import re
import sys

import sqlglot
from sqlglot import exp

HERE = os.path.dirname(os.path.abspath(__file__))
AI_FUNCS = ("ai_filter", "ai_complete")
SUFFIX = re.compile(r"\s*\|\|\s*e'\\n Return a single [^']*'", re.S)  # PLOP's format suffix on every prompt
PASS = ("distinct", "group", "having", "order", "limit")  # select clauses that stay on the rewritten select


def _is_ai(node):
    return isinstance(node, exp.Anonymous) and node.name.lower() in AI_FUNCS


def _has_ai(node):
    return any(_is_ai(n) for n in node.walk())


def _parts(node):
    """Flatten a || chain into its operands."""
    if isinstance(node, exp.DPipe):
        return _parts(node.this) + _parts(node.expression)
    if isinstance(node, exp.Paren):
        return _parts(node.this)
    return [node]


def _split_prompt(call):
    """(question, context expression) of an AI call's prompt: the leading literal up to its last line break
    is the question; what follows (a label such as 'Title: ') opens the context."""
    parts = _parts(call.expressions[0])
    if not parts or not (isinstance(parts[0], exp.Literal) and parts[0].is_string):
        raise ValueError(f"prompt does not start with a literal: {call.sql('duckdb')[:80]}")
    head = parts[0].this
    question, label = (head.rsplit("\n", 1) + [""])[:2] if "\n" in head.strip() else (head, "")
    question = " ".join(question.split())
    rest = list(parts[1:])
    while rest and isinstance(rest[-1], exp.Literal) and rest[-1].is_string and not rest[-1].this.strip():
        rest.pop()
    ctx_parts = ([exp.Literal.string(label.strip() + " ")] if label.strip() else []) + rest
    if not ctx_parts:
        raise ValueError(f"prompt has no context: {call.sql('duckdb')[:80]}")
    ctx = exp.Anonymous(this="concat", expressions=[
        exp.cast(p.copy(), "VARCHAR") if not (isinstance(p, exp.Literal) and p.is_string) else p.copy() for p in ctx_parts])
    return question, ctx


def _conjuncts(node):
    if node is None:
        return []
    if isinstance(node, exp.Where):
        node = node.this
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        return _conjuncts(node.this)
    return [node]


def _colname(tbl, name):
    return f"{tbl}__{name}" if tbl else f"c__{name}"


def _q(s):
    return s.replace("'", "\\'")


def _rewrite_select(select, k, counter):
    """One SELECT with AI calls -> (base name, base select, rewritten select, ingredient texts)."""
    base_name = f"__b{k}"
    # an AI conjunct in an (inner) JOIN ... ON is the same predicate as in WHERE: move it there
    moved = []
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        if on is not None and _has_ai(on):
            keep = [c for c in _conjuncts(on) if not _has_ai(c)]
            moved += [c for c in _conjuncts(on) if _has_ai(c)]
            join.set("on", exp.and_(*keep) if keep else None)
    if moved:
        select.set("where", exp.Where(this=exp.and_(*(_conjuncts(select.args.get("where")) + moved))))

    # the AI calls (an ai_complete int wrapper counts as one call) and their ingredient texts
    ingredients = []  # (placeholder column, ingredient text, prompt column, context expression)
    targets = []      # (node to replace, placeholder)
    seen = set()
    for node in list(select.walk()):
        if id(node) in seen:
            continue
        if isinstance(node, exp.Coalesce) and _has_ai(node):
            call = next(n for n in node.walk() if _is_ai(n))
            q, ctx = _split_prompt(call)
            j = next(counter)
            ingredients.append((f"__ingr{j}", f"COALESCE({{{{LLMMap('{_q(q)}', {base_name}.__p{j}, return_type='int')}}}}, 0)", f"__p{j}", ctx))
            targets.append((node, f"__ingr{j}"))
            seen.update(id(n) for n in node.walk())
        elif _is_ai(node):
            q, ctx = _split_prompt(node)
            j = next(counter)
            if node.name.lower() == "ai_filter":
                text = f"{{{{LLMMap('{_q(q)}', {base_name}.__p{j})}}}} = TRUE"
            else:
                text = f"{{{{LLMMap('{_q(q)}', {base_name}.__p{j}, return_type='str')}}}}"
            ingredients.append((f"__ingr{j}", text, f"__p{j}", ctx))
            targets.append((node, f"__ingr{j}"))
            seen.update(id(n) for n in node.walk())

    # the base select: the relational FROM / JOIN / WHERE, projecting every column the rewritten select
    # reads (renamed table__column) and the rendered prompts
    base = select.copy()
    for key in PASS:
        base.set(key, None)
    rel = [c for c in _conjuncts(base.args.get("where")) if not _has_ai(c)]
    base.set("where", exp.Where(this=exp.and_(*rel)) if rel else None)
    # a star in the select list (SELECT * / t.*) keeps that table's columns under their own names: the base
    # projects the star itself and the rewritten select takes everything but the prompt columns
    star_tables = set()
    for e in select.expressions:
        if isinstance(e, exp.Star):
            star_tables.add("*")
        elif isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
            star_tables.add(e.table or "*")
    all_star = "*" in star_tables

    def starred(tbl):
        return all_star or tbl in star_tables

    cols = {}
    for col in select.find_all(exp.Column):
        if id(col) not in seen and not isinstance(col.this, exp.Star):
            cols.setdefault((col.table or "", col.name), None)
    projections = []
    if all_star:
        projections.append(exp.Star())
    else:
        projections += [exp.column("*", table=t) for t in sorted(star_tables)]
    projections += [exp.alias_(exp.column(name, table=tbl or None), _colname(tbl, name))
                    for tbl, name in cols if not starred(tbl)]
    projections += [exp.alias_(ctx, pcol) for _, _, pcol, ctx in ingredients]
    base.set("expressions", projections)

    # the rewritten select, built fresh: AI nodes -> placeholder columns, columns -> base columns; a bare
    # column in the select list keeps its name for the consumers above
    prompt_cols = [pcol for _, _, pcol, _ in ingredients]
    new_exprs = []
    star_done = False
    for e in select.expressions:
        if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
            if not star_done:
                new_exprs.append(exp.Star(**{"except_": [exp.to_identifier(c) for c in prompt_cols]}))
                star_done = True
        elif isinstance(e, exp.Column):
            new_exprs.append(exp.alias_(e, e.name))
        else:
            new_exprs.append(e)
    select.set("expressions", new_exprs)
    for node, ph in targets:
        node.replace(exp.column(ph, table=base_name))
    for col in list(select.find_all(exp.Column)):
        key = (col.table or "", col.name)
        if col.table != base_name and key in cols:
            col.replace(exp.column(key[1] if starred(key[0]) else _colname(*key), table=base_name))
    ai_conj = [c for c in _conjuncts(select.args.get("where"))
               if any(n.name.startswith("__ingr") for n in c.find_all(exp.Column))]
    new = exp.select(*[e for e in select.expressions]).from_(base_name)
    if ai_conj:
        new = new.where(exp.and_(*ai_conj))
    for key in PASS:
        if select.args.get(key) is not None:
            new.set(key, select.args[key])
    return base_name, base, new, ingredients


def translate(sql):
    """-> (setup statements for the DuckDB backend, BlendSQL query text)."""
    sql = SUFFIX.sub("", sql)
    statements = sqlglot.parse(sql, read="duckdb")
    setup = [s.sql("duckdb") for s in statements[:-1]]
    query = statements[-1]
    counter = iter(range(1000))
    with_ = query.args.get("with_")
    ctes = [(cte.alias, cte.this) for cte in (with_.expressions if with_ is not None else [])]
    query.set("with_", None)
    # a derived table holding an AI call becomes a CTE (BlendSQL evaluates ingredients in CTEs and the top level)
    hoisted = 0
    for sub in list(query.find_all(exp.Subquery)):
        if isinstance(sub.this, exp.Select) and _has_ai(sub.this) and isinstance(sub.parent, (exp.From, exp.Join)):
            hoisted += 1
            name = f"__d{hoisted}"
            ctes.append((name, sub.this))
            sub.replace(exp.alias_(exp.to_table(name), sub.alias or name, table=True))
    ingredients, out_ctes, k = [], [], 0
    for name, body in ctes:
        if isinstance(body, exp.Select) and _has_ai(body):
            k += 1
            base_name, base, new, ingr = _rewrite_select(body, k, counter)
            ingredients += ingr
            out_ctes += [(base_name, base), (name, new)]
        else:
            out_ctes.append((name, body))
    if _has_ai(query):
        k += 1
        base_name, base, new, ingr = _rewrite_select(query, k, counter)
        ingredients += ingr
        out_ctes.append((base_name, base))
        query = new
    for name, body in out_ctes:
        query = query.with_(name, as_=body, append=True)
    text = query.sql("duckdb", pretty=True)
    for ph, ingr_text, _, _ in ingredients:
        text = re.sub(rf"\b__b\d+\.{ph}\b|\b{ph}\b", lambda m: ingr_text, text)
    return setup, text


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    os.makedirs(os.path.join(HERE, "blendsql_queries"), exist_ok=True)
    for name in which:
        sql = open(os.path.join(HERE, "swan_queries", f"{name}.sql")).read()
        try:
            setup, text = translate(sql)
        except Exception as ex:  # noqa: BLE001
            print(f"{name}: FAILED {type(ex).__name__}: {ex}")
            continue
        out = "-- setup (DuckDB backend)\n" + ";\n".join(setup) + ";\n-- query (BlendSQL)\n" + text + "\n"
        if sys.argv[1:]:
            print(out)
        open(os.path.join(HERE, "blendsql_queries", f"{name}.sql"), "w").write(out)
        print(f"{name}: ok ({text.count('LLMMap')} ingredients)")


if __name__ == "__main__":
    main()
