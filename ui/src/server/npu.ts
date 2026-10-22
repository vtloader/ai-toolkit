import { execFile } from 'child_process';
import { promisify } from 'util';
import { NpuInfo } from '@/types';

const execFileAsync = promisify(execFile);
const NPU_SMI_TIMEOUT_MS = 5000;

export interface NpuStats {
  available: boolean;
  driverVersion: string;
  npus: NpuInfo[];
  error?: string;
}

let driverVersionPromise: Promise<string> | null = null;
let logicIdsPromise: Promise<Map<string, number>> | null = null;

/**
 * `npu-smi -v` prints e.g. "npu-smi version: 25.5.5". Cached because the
 * monitor only needs the driver version once; if the tool is missing every
 * caller gets "unknown" without spawning another child.
 */
function readDriverVersion(): Promise<string> {
  if (!driverVersionPromise) {
    driverVersionPromise = execFileAsync('npu-smi', ['-v'], { timeout: NPU_SMI_TIMEOUT_MS })
      .then(({ stdout }) => {
        const match = stdout.match(/(?:npu-smi\s+version|Version)\s*[:：]\s*(\S+)/i);
        return match?.[1] ?? 'unknown';
      })
      .catch(() => 'unknown');
  }
  return driverVersionPromise;
}

function toNumber(value: string): number {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * `npu-smi info -m` maps card/chip ids to the Chip Logic ID that
 * ASCEND_RT_VISIBLE_DEVICES and torch.npu use. The first line is the header;
 * rows without a numeric logic id (an MCU row) are skipped.
 */
export function parseNpuLogicIds(text: string): Map<string, number> {
  const ids = new Map<string, number>();
  const pattern = /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+/;
  for (const line of text.split('\n')) {
    const match = line.match(pattern);
    if (match) {
      ids.set(`${match[1]}:${match[2]}`, parseInt(match[3], 10));
    }
  }
  return ids;
}

function readLogicIds(): Promise<Map<string, number>> {
  if (!logicIdsPromise) {
    logicIdsPromise = execFileAsync('npu-smi', ['info', '-m'], { timeout: NPU_SMI_TIMEOUT_MS })
      .then(({ stdout }) => parseNpuLogicIds(stdout))
      .catch(() => new Map());
  }
  return logicIdsPromise;
}

/**
 * Parse the fixed-width `npu-smi info` table. Each chip spans two rows:
 *
 *   | 3     Ascend910           | OK            | 175.5       45                0    / 0             |
 *   | 0     6                   | 0000:0A:00.0  | 0           0    / 0          3119 / 65536         |
 *
 * The first row carries the card id/name/health/power/temperature, the second
 * the chip id/physical id/bus/AICore/HBM figures. Header and process rows do
 * not match either shape and are skipped. A card may have several chips.
 */
export function parseNpuSmiInfo(text: string, driverVersion = 'unknown'): NpuInfo[] {
  const cardPattern =
    /^\|\s*(\d+)\s+([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([-\d.]+)\s+([-\d.]+)\s+(\d+)\s*\/\s*(\d+)\s*\|\s*$/;
  const chipPattern =
    /^\|\s*(\d+)\s+(\d+)\s*\|\s*([^|]+?)\s*\|\s*([-\d.]+)\s+(\d+)\s*\/\s*(\d+)\s+(\d+)\s*\/\s*(\d+)\s*\|\s*$/;

  const npus: NpuInfo[] = [];
  let card: { id: number; name: string; health: string; power: number; temperature: number } | null = null;

  for (const line of text.split('\n')) {
    const cardMatch = line.match(cardPattern);
    if (cardMatch) {
      card = {
        id: parseInt(cardMatch[1], 10),
        name: cardMatch[2].trim(),
        health: cardMatch[3].trim(),
        power: toNumber(cardMatch[4]),
        temperature: toNumber(cardMatch[5]),
      };
      continue;
    }

    const chipMatch = line.match(chipPattern);
    if (!chipMatch || !card) continue;

    const total = toNumber(chipMatch[8]);
    const used = toNumber(chipMatch[7]);
    npus.push({
      index: npus.length,
      cardId: card.id,
      chipId: parseInt(chipMatch[1], 10),
      logicId: parseInt(chipMatch[1], 10),
      name: card.name,
      driverVersion,
      health: card.health,
      busId: chipMatch[3].trim(),
      temperature: card.temperature,
      utilization: {
        aicore: toNumber(chipMatch[4]),
        memory: total > 0 ? Math.round((used / total) * 100) : 0,
      },
      memory: {
        total,
        used,
        free: Math.max(0, total - used),
      },
      power: {
        draw: card.power,
        limit: 0,
      },
    });
  }

  return npus;
}

/**
 * One-shot sample of every Ascend NPU visible to npu-smi. Returns
 * `available: false` when the tool is not installed (the common CUDA-only
 * case) so callers can report it without treating it as an error.
 */
export async function getNpuStats(): Promise<NpuStats> {
  const [driverVersion, logicIds] = await Promise.all([readDriverVersion(), readLogicIds()]);
  try {
    const { stdout } = await execFileAsync('npu-smi', ['info'], {
      timeout: NPU_SMI_TIMEOUT_MS,
      maxBuffer: 4 * 1024 * 1024,
    });
    const npus = parseNpuSmiInfo(stdout, driverVersion).map(npu => ({
      ...npu,
      logicId: logicIds.get(`${npu.cardId}:${npu.chipId}`) ?? npu.chipId,
    }));
    return {
      available: true,
      driverVersion,
      npus,
    };
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return {
      available: false,
      driverVersion,
      npus: [],
      error:
        code === 'ENOENT'
          ? 'npu-smi not found or not accessible'
          : `Failed to fetch NPU stats: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}
