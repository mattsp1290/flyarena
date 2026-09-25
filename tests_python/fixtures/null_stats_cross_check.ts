// Tiny CLI shim for `tests_python/test_null_stats_cross_check.py`: reads a
// JSON object of test cases from stdin, runs them through the REAL
// `scripts/null/null-stats.ts` `nullSummary`/`rankStatistics` (not a
// reimplementation), and writes the results as JSON to stdout. Run via
// `node_modules/.bin/tsx` -- no build step, no test framework, just this
// file's own process boundary.
//
// This exists so the Python-side pytest can assert Python's
// `explain_stats.null_range_summary`/`rank_statistics` against the TS
// implementation's *actual* output, not hand-copied expected values (a
// thermo-maintainability review finding: the previous test only checked
// Python against itself).

import { nullSummary, rankStatistics } from '../../scripts/null/null-stats';

interface Input {
  readonly nullSummaryCases: ReadonlyArray<readonly number[]>;
  readonly rankStatisticsCases: ReadonlyArray<{ readonly nullValues: readonly number[]; readonly bioValue: number }>;
}

const readStdin = async (): Promise<string> => {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
};

const main = async (): Promise<void> => {
  const raw = await readStdin();
  const input = JSON.parse(raw) as Input;

  const nullSummaryResults = input.nullSummaryCases.map((values) => {
    const summary = nullSummary(values);
    return { median: summary.median, p2_5: summary.p2_5, p97_5: summary.p97_5 };
  });

  const rankStatisticsResults = input.rankStatisticsCases.map(({ nullValues, bioValue }) => {
    const result = rankStatistics(nullValues, bioValue);
    return { kBelow: result.kBelow, kEqual: result.kEqual, bioPercentile: result.bioPercentile };
  });

  process.stdout.write(JSON.stringify({ nullSummaryResults, rankStatisticsResults }));
};

void main();
