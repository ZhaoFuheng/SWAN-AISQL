CREATE TABLE govreport AS SELECT id, summary, source FROM read_csv('govreport/govreport_summary.csv', header=true);
