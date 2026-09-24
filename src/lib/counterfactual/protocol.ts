import type { AtlasSelection } from '../atlas/types';
import type { GraphMode } from '../connectome/format';
import type { ExportDocument, Preparation, Request } from './types';

export type WorkerCommand =
  | { type: 'prepare'; baseUrl: string; topology: GraphMode }
  | { type: 'run-atlas'; baseUrl: string; selection: AtlasSelection; request: Request }
  | { type: 'run'; baseUrl: string; request: Request };
export type WorkerEvent =
  | { type: 'prepared'; preparation: Preparation }
  | { type: 'progress'; completed: number; total: number }
  | { type: 'complete'; document: ExportDocument }
  | { type: 'error'; message: string };
