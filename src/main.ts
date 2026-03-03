import { startParticleLife } from './particleLife/main';
import { startCellSim }      from './cellSim/main';

const canvas     = document.getElementById('canvas')     as HTMLCanvasElement;
const uiDiv      = document.getElementById('ui')         as HTMLDivElement;
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;

canvas.width  = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
});

type StopFn = () => void;
let stopCurrent: StopFn | null = null;

async function switchMode(mode: string): Promise<void> {
  // Stop the current simulation
  if (stopCurrent) {
    stopCurrent();
    stopCurrent = null;
  }
  // Clear the UI panel
  uiDiv.innerHTML = '';

  // Start the selected simulation
  if (mode === 'particleLife') {
    stopCurrent = await startParticleLife(canvas, uiDiv);
  } else if (mode === 'cellSim') {
    stopCurrent = await startCellSim(canvas, uiDiv);
  }
}

modeSelect.addEventListener('change', () => {
  switchMode(modeSelect.value).catch((err: unknown) => {
    console.error(err);
  });
});

// Start with Particle Life by default
switchMode('particleLife').catch((err: unknown) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:red;padding:20px">${err}</pre>`;
});
