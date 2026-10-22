import React from 'react';
import { Cpu, HardDrive, Layers, Thermometer, Zap } from 'lucide-react';
import { NpuInfo } from '@/types';

interface NPUWidgetProps {
  npu: NpuInfo;
}

export default function NPUWidget({ npu }: NPUWidgetProps) {
  const formatMemory = (mb: number): string => {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const getUtilizationColor = (value: number): string => {
    return value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';
  };

  const getTemperatureColor = (temp: number): string => {
    return temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';
  };

  const healthColor =
    npu.health.toUpperCase() === 'OK'
      ? 'bg-emerald-900/60 text-emerald-300'
      : 'bg-amber-900/60 text-amber-300';
  const memoryPercent = npu.memory.total > 0 ? (npu.memory.used / npu.memory.total) * 100 : 0;

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <h2 className="font-semibold text-gray-100">{npu.name}</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">
            NPU {npu.cardId}:{npu.chipId}
          </span>
        </div>
        <span className={`px-2 py-0.5 rounded-full text-xs ${healthColor}`}>{npu.health || 'Unknown'}</span>
      </div>

      <div className="p-4 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Thermometer className={`w-4 h-4 ${getTemperatureColor(npu.temperature)}`} />
              <div>
                <p className="text-xs text-gray-400">Temperature</p>
                <p className={`text-sm font-medium ${getTemperatureColor(npu.temperature)}`}>
                  {npu.temperature}°C
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Chip</p>
                <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                  Card {npu.cardId} / Chip {npu.chipId}
                </p>
              </div>
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2 mb-1">
              <Cpu className="w-4 h-4 text-gray-400" />
              <p className="text-xs text-gray-400">AICore Load</p>
              <span className="text-xs text-gray-300 ml-auto">{npu.utilization.aicore}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className={`h-1 rounded-full transition-all ${getUtilizationColor(npu.utilization.aicore)}`}
                style={{ width: `${npu.utilization.aicore}%` }}
              />
            </div>
            <div className="flex items-center space-x-2 mb-1 mt-3">
              <HardDrive className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <p className="text-xs text-gray-400">HBM</p>
              <span className="text-xs text-gray-300 ml-auto">{memoryPercent.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-1">
              <div
                className="h-1 rounded-full bg-blue-500 transition-all"
                style={{ width: `${memoryPercent}%` }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              {formatMemory(npu.memory.used)} / {formatMemory(npu.memory.total)}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-gray-800">
          <div className="flex items-start space-x-2">
            <Zap className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            <div>
              <p className="text-xs text-gray-400">Power Draw</p>
              <p className="text-sm text-gray-200">
                {npu.power.draw > 0 ? `${npu.power.draw.toFixed(1)}W` : 'N/A'}
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-2">
            <Layers className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            <div>
              <p className="text-xs text-gray-400">Driver</p>
              <p className="text-sm text-gray-200">{npu.driverVersion}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
