import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {
  Bot,
  Circle,
  Disc3,
  Mic,
  Pause,
  Piano,
  Play,
  Plus,
  Radio,
  RotateCcw,
  Square,
  Trash2,
  Wand2,
  Waves,
} from 'lucide-react';
import './styles.css';

type TrackKind = 'audio' | 'midi';

type Track = {
  id: string;
  name: string;
  kind: TrackKind;
  color: string;
  bars: number;
  sourceBuffer: AudioBuffer | null;
  transformedBuffer: AudioBuffer | null;
  playbackMode: 'source' | 'transformed';
  promptStrength: number;
  initNoiseLevel: number;
  drumless: boolean;
  doubleGenerate: boolean;
  smoothFade: boolean;
  volume: number;
  active: boolean;
  armed: boolean;
  prompt: string;
  notes: MidiNote[];
};

type MidiNote = {
  note: number;
  velocity: number;
  start: number;
  duration: number;
};

type RecordState =
  | {mode: 'idle'}
  | {mode: 'count-in'; target: 'audio' | 'midi'; trackId: string; remaining: number}
  | {mode: 'recording'; target: 'audio' | 'midi'; trackId: string; remaining: number};

const COLORS = ['#ff5c39', '#ffe45e', '#52d273', '#4cb7ff', '#ff7ad9', '#f5f0e6'];
const DEFAULT_PROMPTS = [
  'rubbery analog bass, tight tape loop',
  'sparkling mallet synths with dusty drums',
  'motorik krautrock pulse, warm cassette',
  'glitchy teenage engineering pocket orchestra',
];

function makeTrack(index: number, kind: TrackKind = 'audio'): Track {
  return {
    id: crypto.randomUUID(),
    name: `LOOP ${index + 1}`,
    kind,
    color: COLORS[index % COLORS.length],
    bars: index === 2 ? 4 : 2,
    sourceBuffer: null,
    transformedBuffer: null,
    playbackMode: 'source',
    promptStrength: 3,
    initNoiseLevel: 0.5,
    drumless: false,
    doubleGenerate: false,
    smoothFade: false,
    volume: 0.8,
    active: true,
    armed: index === 0,
    prompt: DEFAULT_PROMPTS[index % DEFAULT_PROMPTS.length],
    notes: [],
  };
}

function getAudioContext(): AudioContext {
  const AudioCtor = window.AudioContext || (window as typeof window & {webkitAudioContext: typeof AudioContext}).webkitAudioContext;
  const context = (window as typeof window & {__loopAudio?: AudioContext}).__loopAudio;
  if (context) return context;
  const created = new AudioCtor({sampleRate: 48000});
  (window as typeof window & {__loopAudio?: AudioContext}).__loopAudio = created;
  return created;
}

async function decodeBlob(blob: Blob): Promise<AudioBuffer> {
  const context = getAudioContext();
  const bytes = await blob.arrayBuffer();
  return await context.decodeAudioData(bytes.slice(0));
}

async function normalizeLoopBuffer(input: AudioBuffer, seconds: number): Promise<AudioBuffer> {
  const sampleRate = input.sampleRate;
  const length = Math.max(1, Math.round(seconds * sampleRate));
  const offline = new OfflineAudioContext(2, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = input;
  source.connect(offline.destination);
  source.start(0);
  return await offline.startRendering();
}

async function prepareGeneratedLoop(
  input: AudioBuffer,
  loopSeconds: number,
  doubleGenerate: boolean,
  smoothFade: boolean,
  cropStartFrames?: number,
): Promise<AudioBuffer> {
  const sampleRate = input.sampleRate;
  const loopLength = Math.max(1, Math.round(loopSeconds * sampleRate));
  const start = cropStartFrames === undefined
    ? (doubleGenerate ? loopLength : 0)
    : Math.max(0, cropStartFrames);
  const output = new AudioBuffer({
    length: loopLength,
    numberOfChannels: Math.min(2, input.numberOfChannels),
    sampleRate,
  });

  for (let channel = 0; channel < output.numberOfChannels; channel++) {
    const src = input.getChannelData(channel);
    const dst = output.getChannelData(channel);
    for (let i = 0; i < loopLength; i++) {
      dst[i] = src[Math.min(src.length - 1, start + i)] ?? 0;
    }

    if (smoothFade && start > 0) {
      const fadeLength = Math.min(Math.round(0.5 * sampleRate), loopLength, start);
      for (let i = 0; i < fadeLength; i++) {
        const t = i / Math.max(1, fadeLength - 1);
        const dstIndex = loopLength - fadeLength + i;
        const leadInIndex = start - fadeLength + i;
        const leadIn = src[Math.max(0, Math.min(src.length - 1, leadInIndex))] ?? 0;
        dst[dstIndex] = dst[dstIndex] * (1 - t) + leadIn * t;
      }
    } else if (smoothFade) {
      const fadeLength = Math.min(Math.round(0.5 * sampleRate), loopLength);
      for (let i = 0; i < fadeLength; i++) {
        const t = i / Math.max(1, fadeLength - 1);
        const dstIndex = loopLength - fadeLength + i;
        const head = dst[i];
        dst[dstIndex] = dst[dstIndex] * (1 - t) + head * t;
      }
    }
  }

  return output;
}

async function resampleLoopToSampleGrid(input: AudioBuffer, loopSeconds: number, sampleRate: number): Promise<AudioBuffer> {
  const loopFrames = Math.max(1, Math.round(loopSeconds * sampleRate));
  const offline = new OfflineAudioContext(2, loopFrames, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = input;
  source.loop = true;
  source.loopEnd = Math.min(loopSeconds, input.duration);
  source.connect(offline.destination);
  source.start(0);
  source.stop(loopFrames / sampleRate);
  return await offline.startRendering();
}

function getStableAudioGenerationPlan(loopSeconds: number, sampleRate = 44100) {
  const safeLoopSeconds = Math.max(0.1, loopSeconds);
  const maxRequestSeconds = 65;
  const loopFrames = Math.max(1, Math.round(safeLoopSeconds * sampleRate));
  const minWarmupFrames = Math.ceil(20 * sampleRate);
  const minRequestFrames = Math.ceil(30 * sampleRate);
  const maxRequestFrames = Math.floor(maxRequestSeconds * sampleRate);
  const warmupRepeats = Math.max(1, Math.ceil(minWarmupFrames / loopFrames));
  const desiredCropFrames = warmupRepeats * loopFrames;
  const desiredRequestFrames = Math.max(minRequestFrames, desiredCropFrames + loopFrames * 2);
  const requestFrames = Math.min(maxRequestFrames, desiredRequestFrames);
  const cropFrames = Math.min(desiredCropFrames, Math.max(0, requestFrames - loopFrames));
  return {
    cropFrames,
    cropStart: cropFrames / sampleRate,
    loopFrames,
    requestFrames,
    requestSeconds: requestFrames / sampleRate,
    sampleRate,
    warmupRepeats,
  };
}

async function renderRepeatedAudioInput(input: AudioBuffer, plan: ReturnType<typeof getStableAudioGenerationPlan>): Promise<AudioBuffer> {
  const loop = await resampleLoopToSampleGrid(input, plan.loopFrames / plan.sampleRate, plan.sampleRate);
  const output = new AudioBuffer({
    length: plan.requestFrames,
    numberOfChannels: 2,
    sampleRate: plan.sampleRate,
  });

  for (let channel = 0; channel < output.numberOfChannels; channel++) {
    const src = loop.getChannelData(Math.min(channel, loop.numberOfChannels - 1));
    const dst = output.getChannelData(channel);
    for (let i = 0; i < output.length; i++) {
      dst[i] = src[i % plan.loopFrames] ?? 0;
    }
  }

  return output;
}

function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const channelCount = Math.min(2, buffer.numberOfChannels);
  const frameCount = buffer.length;
  const dataBytes = frameCount * 2 * 2;
  const bytes = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(bytes);
  const channels = [
    buffer.getChannelData(0),
    buffer.getChannelData(channelCount > 1 ? 1 : 0),
  ];

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2 * 2, true);
  view.setUint16(32, 2 * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < frameCount; i++) {
    for (let channel = 0; channel < 2; channel++) {
      const sample = clamp(channels[channel][i] ?? 0, -1, 1);
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([bytes], {type: 'audio/wav'});
}

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function renderMidiNotes(notes: MidiNote[], seconds: number): Promise<AudioBuffer> {
  const sampleRate = 48000;
  const offline = new OfflineAudioContext(2, Math.round(seconds * sampleRate), sampleRate);
  const master = offline.createGain();
  master.gain.value = 0.75;
  master.connect(offline.destination);

  notes.forEach((event) => {
    const freq = 440 * 2 ** ((event.note - 69) / 12);
    const osc = offline.createOscillator();
    const sub = offline.createOscillator();
    const filter = offline.createBiquadFilter();
    const amp = offline.createGain();

    osc.type = 'sawtooth';
    sub.type = 'square';
    osc.frequency.value = freq;
    sub.frequency.value = freq / 2;
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(380 + event.velocity * 2600, event.start);
    filter.Q.value = 6;
    amp.gain.setValueAtTime(0.0001, event.start);
    amp.gain.exponentialRampToValueAtTime(0.18 * event.velocity, event.start + 0.012);
    amp.gain.exponentialRampToValueAtTime(0.0001, Math.min(seconds, event.start + event.duration + 0.18));

    osc.connect(filter);
    sub.connect(filter);
    filter.connect(amp);
    amp.connect(master);
    osc.start(event.start);
    sub.start(event.start);
    osc.stop(Math.min(seconds, event.start + event.duration + 0.22));
    sub.stop(Math.min(seconds, event.start + event.duration + 0.22));
  });

  return await offline.startRendering();
}

function barsToSeconds(bars: number, bpm: number) {
  return bars * 4 * (60 / bpm);
}

function getPlayableBuffer(track: Track) {
  if (track.playbackMode === 'transformed' && track.transformedBuffer) {
    return track.transformedBuffer;
  }
  return track.sourceBuffer;
}

function getTrackSeconds(track: Track, bpm: number) {
  return barsToSeconds(track.bars, bpm);
}

function makeWaveformPath(buffer: AudioBuffer, width: number, height: number, points = 72) {
  const channel = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(channel.length / points));
  const mid = height / 2;
  const coords: string[] = [];
  for (let i = 0; i < points; i++) {
    let peak = 0;
    const start = i * step;
    const end = Math.min(channel.length, start + step);
    for (let j = start; j < end; j++) peak = Math.max(peak, Math.abs(channel[j]));
    const x = (i / (points - 1)) * width;
    const y = mid - peak * (height * 0.42);
    coords.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  for (let i = points - 1; i >= 0; i--) {
    let peak = 0;
    const start = i * step;
    const end = Math.min(channel.length, start + step);
    for (let j = start; j < end; j++) peak = Math.max(peak, Math.abs(channel[j]));
    const x = (i / (points - 1)) * width;
    const y = mid + peak * (height * 0.42);
    coords.push(`L ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  return `${coords.join(' ')} Z`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function repeatMidiNotes(notes: MidiNote[], loopSeconds: number, totalSeconds: number) {
  const repeats = Math.max(1, Math.ceil(totalSeconds / loopSeconds));
  const result: MidiNote[] = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    for (const note of notes) {
      const start = note.start + repeat * loopSeconds;
      if (start >= totalSeconds) continue;
      result.push({
        ...note,
        start,
        duration: Math.min(note.duration, Math.max(0.05, totalSeconds - start)),
      });
    }
  }
  return result;
}

function CaptureGraphic({track, bpm}: {track: Track; bpm: number}) {
  const width = 360;
  const height = 54;
  const source = track.sourceBuffer;
  if (!source) {
    return (
      <div className="capture-graphic empty">
        <Waves size={18} />
      </div>
    );
  }

  if (track.kind === 'midi') {
    const seconds = getTrackSeconds(track, bpm);
    const notes = track.notes.length ? track.notes : [];
    const minNote = Math.min(...notes.map((note) => note.note), 48);
    const maxNote = Math.max(...notes.map((note) => note.note), 84);
    const noteRange = Math.max(1, maxNote - minNote);
    return (
      <svg className="capture-graphic" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
        <rect x="0" y="0" width={width} height={height} />
        {notes.map((note, index) => {
          const x = (note.start / seconds) * width;
          const w = Math.max(4, (note.duration / seconds) * width);
          const y = height - 8 - ((note.note - minNote) / noteRange) * (height - 16);
          return <rect key={`${note.note}-${index}`} className="midi-note" x={x} y={y} width={w} height="5" rx="1" />;
        })}
      </svg>
    );
  }

  return (
    <svg className="capture-graphic" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <rect x="0" y="0" width={width} height={height} />
      <path className="waveform" d={makeWaveformPath(source, width, height)} />
    </svg>
  );
}

function App() {
  const initialTracksRef = useRef<Track[] | null>(null);
  if (!initialTracksRef.current) {
    initialTracksRef.current = [0, 1, 2, 3].map((i) => makeTrack(i, i === 1 ? 'midi' : 'audio'));
  }
  const [tracks, setTracks] = useState<Track[]>(initialTracksRef.current);
  const [bpm, setBpm] = useState(112);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const [recordState, setRecordState] = useState<RecordState>({mode: 'idle'});
  const [status, setStatus] = useState('Backend idle. Record locally or generate a Magenta loop.');
  const [model, setModel] = useState('mrt2_small');
  const [serverModels, setServerModels] = useState<string[]>([]);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [midiReady, setMidiReady] = useState(false);
  const [metronomeEnabled, setMetronomeEnabled] = useState(false);
  const [activeSynthNotes, setActiveSynthNotes] = useState<Set<number>>(new Set());

  const beatSeconds = 60 / bpm;
  const startedAtRef = useRef(0);
  const sourcesRef = useRef<Map<string, {source: AudioBufferSourceNode; gain: GainNode}>>(new Map());
  const tracksRef = useRef<Track[]>(initialTracksRef.current);
  const bpmRef = useRef(bpm);
  const isPlayingRef = useRef(false);
  const metronomeEnabledRef = useRef(false);
  const lastMetronomeBeatRef = useRef(-1);
  const timersRef = useRef<number[]>([]);
  const midiNotesRef = useRef<MidiNote[]>([]);
  const midiHeldRef = useRef<Map<number, {velocity: number; start: number}>>(new Map());
  const liveOscRef = useRef<Map<number, {osc: OscillatorNode; gain: GainNode}>>(new Map());
  const recordStateRef = useRef<RecordState>({mode: 'idle'});

  const selectedTrack = tracks.find((track) => track.id === selectedTrackId) ?? tracks[0];
  const elapsedBeats = playhead / beatSeconds;
  const beatProgress = elapsedBeats % 4;
  const progressDegrees = (beatProgress / 4) * 360;
  const visibleBeat = Math.floor(beatProgress) + 1;

  useEffect(() => {
    recordStateRef.current = recordState;
  }, [recordState]);

  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  useEffect(() => {
    bpmRef.current = bpm;
  }, [bpm]);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    metronomeEnabledRef.current = metronomeEnabled;
    lastMetronomeBeatRef.current = -1;
  }, [metronomeEnabled]);

  const clearTimers = () => {
    timersRef.current.forEach((timer) => window.clearTimeout(timer));
    timersRef.current = [];
  };

  const stopSources = useCallback(() => {
    sourcesRef.current.forEach(({source}) => {
      try {
        source.stop();
      } catch {
        // Already stopped.
      }
    });
    sourcesRef.current.clear();
  }, []);

  const playMetronomeClick = useCallback((accent: boolean) => {
    if (!metronomeEnabledRef.current) return;
    const context = getAudioContext();
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.type = 'square';
    osc.frequency.value = accent ? 1760 : 1180;
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(accent ? 0.22 : 0.14, context.currentTime + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.055);
    osc.connect(gain);
    gain.connect(context.destination);
    osc.start();
    osc.stop(context.currentTime + 0.065);
  }, []);

  const startSourcesForTracks = useCallback((trackList: Track[], when = 0, alignToTransport = true) => {
    const context = getAudioContext();
    stopSources();
    const elapsed = Math.max(0, context.currentTime - startedAtRef.current);
    const scheduledDelay = Math.max(0, when - context.currentTime);
    const elapsedAtStart = elapsed + scheduledDelay;
    trackList.forEach((track) => {
      const buffer = getPlayableBuffer(track);
      if (!track.active || !buffer) return;
      const loopSeconds = getTrackSeconds(track, bpmRef.current);
      const offset = alignToTransport && loopSeconds > 0 ? elapsedAtStart % loopSeconds : 0;
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = buffer;
      source.loop = true;
      source.loopEnd = loopSeconds;
      gain.gain.value = track.volume;
      source.connect(gain);
      gain.connect(context.destination);
      source.start(when, offset);
      sourcesRef.current.set(track.id, {source, gain});
    });
  }, [stopSources]);

  const syncRunningSources = useCallback((trackList: Track[]) => {
    if (!isPlayingRef.current) return;
    startSourcesForTracks(trackList, getAudioContext().currentTime + 0.025, true);
  }, [startSourcesForTracks]);

  useEffect(() => {
    bpmRef.current = bpm;
    syncRunningSources(tracksRef.current);
  }, [bpm, syncRunningSources]);

  const updateTracks = useCallback((updater: (current: Track[]) => Track[], syncAudio = true) => {
    setTracks((current) => {
      const next = updater(current);
      tracksRef.current = next;
      if (syncAudio) syncRunningSources(next);
      return next;
    });
  }, [syncRunningSources]);

  const startPlayback = useCallback(async () => {
    const context = getAudioContext();
    await context.resume();
    const when = context.currentTime + 0.045;
    startedAtRef.current = when;
    lastMetronomeBeatRef.current = -1;
    startSourcesForTracks(tracksRef.current, when, false);
    isPlayingRef.current = true;
    setIsPlaying(true);
  }, [startSourcesForTracks]);

  const stopPlayback = useCallback(() => {
    stopSources();
    isPlayingRef.current = false;
    lastMetronomeBeatRef.current = -1;
    setIsPlaying(false);
    setPlayhead(0);
  }, [stopSources]);

  useEffect(() => {
    sourcesRef.current.forEach((node, trackId) => {
      const track = tracks.find((candidate) => candidate.id === trackId);
      if (track) node.gain.gain.value = track.volume;
    });
  }, [tracks]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (!isPlaying && recordState.mode === 'idle') return;
      const context = getAudioContext();
      if (isPlaying) {
        const elapsed = Math.max(0, context.currentTime - startedAtRef.current);
        const beatIndex = Math.floor(elapsed / beatSeconds);
        if (metronomeEnabledRef.current && beatIndex !== lastMetronomeBeatRef.current) {
          lastMetronomeBeatRef.current = beatIndex;
          playMetronomeClick(beatIndex % 4 === 0);
        }
        setPlayhead(elapsed);
      }
    }, 33);
    return () => window.clearInterval(interval);
  }, [beatSeconds, isPlaying, playMetronomeClick, recordState.mode]);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then((data) => {
        if (Array.isArray(data.localModels)) {
          setServerModels(data.localModels);
          if (data.localModels.includes('mrt2_small')) setModel('mrt2_small');
          else if (data.localModels[0]) setModel(data.localModels[0]);
        }
      })
      .catch(() => setStatus('Backend not running yet. Start server/looper_server.py for Magenta generation.'));
  }, []);

  const setTrackSourceBuffer = useCallback((trackId: string, buffer: AudioBuffer, patch: Partial<Track> = {}) => {
    updateTracks((current) =>
      current.map((track) =>
        track.id === trackId
          ? {...track, sourceBuffer: buffer, transformedBuffer: null, playbackMode: 'source', active: true, ...patch}
          : track,
      ),
    );
  }, [updateTracks]);

  const setTrackTransformedBuffer = useCallback((trackId: string, buffer: AudioBuffer, patch: Partial<Track> = {}) => {
    updateTracks((current) =>
      current.map((track) =>
        track.id === trackId
          ? {...track, transformedBuffer: buffer, playbackMode: 'transformed', active: true, ...patch}
          : track,
      ),
    );
  }, [updateTracks]);

  const trackDuration = (trackId: string) => {
    const track = tracksRef.current.find((candidate) => candidate.id === trackId);
    return track ? getTrackSeconds(track, bpm) : barsToSeconds(2, bpm);
  };

  const countInMs = () => {
    if (!isPlaying) return beatSeconds * 1000;
    const context = getAudioContext();
    const elapsed = Math.max(0, context.currentTime - startedAtRef.current);
    const beatPos = elapsed / beatSeconds;
    const remainder = 4 - (beatPos % 4);
    return (remainder < 0.08 ? 4 : remainder) * beatSeconds * 1000;
  };

  const scheduleCountInClicks = (waitMs: number) => {
    if (!metronomeEnabledRef.current) return;
    const beats = Math.max(1, Math.ceil(waitMs / (beatSeconds * 1000)));
    for (let i = 0; i < beats; i++) {
      const delay = Math.max(0, waitMs - (beats - i) * beatSeconds * 1000);
      timersRef.current.push(window.setTimeout(() => playMetronomeClick(i === 0), delay));
    }
  };

  const updateCountdown = (target: 'audio' | 'midi', trackId: string, startMs: number, durationMs: number) => {
    const tick = window.setInterval(() => {
      const now = performance.now();
      if (now < startMs) {
        setRecordState({mode: 'count-in', target, trackId, remaining: Math.max(0, (startMs - now) / 1000)});
      } else if (now < startMs + durationMs) {
        setRecordState({mode: 'recording', target, trackId, remaining: Math.max(0, (startMs + durationMs - now) / 1000)});
      } else {
        window.clearInterval(tick);
      }
    }, 50);
    timersRef.current.push(tick);
  };

  const recordAudio = async (trackId: string) => {
    clearTimers();
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const recorder = new MediaRecorder(stream);
    const chunks: BlobPart[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const decoded = await decodeBlob(new Blob(chunks, {type: recorder.mimeType}));
      const seconds = trackDuration(trackId);
      const loop = await normalizeLoopBuffer(decoded, seconds);
      setTrackSourceBuffer(trackId, loop, {kind: 'audio', notes: []});
      clearTimers();
      setRecordState({mode: 'idle'});
      setStatus(`Captured ${seconds.toFixed(1)}s audio loop.`);
    };

    const wait = countInMs();
    const startMs = performance.now() + wait;
    const durationMs = trackDuration(trackId) * 1000;
    setStatus('Audio armed. Recording starts on the next bar edge.');
    updateCountdown('audio', trackId, startMs, durationMs);
    scheduleCountInClicks(wait);
    timersRef.current.push(window.setTimeout(() => recorder.start(), wait));
    timersRef.current.push(window.setTimeout(() => recorder.stop(), wait + durationMs));
  };

  const ensureMidi = async () => {
    if (!navigator.requestMIDIAccess) {
      setStatus('This browser does not expose Web MIDI. Chrome is the best target for MIDI capture.');
      return false;
    }
    const access = await navigator.requestMIDIAccess();
    access.inputs.forEach((input) => {
      input.onmidimessage = (event) => handleMidiMessage(event);
    });
    setMidiReady(true);
    setStatus(`MIDI ready: ${access.inputs.size || 0} input(s).`);
    return true;
  };

  const noteOn = (note: number, velocity: number) => {
    const context = getAudioContext();
    const osc = context.createOscillator();
    const gain = context.createGain();
    osc.type = 'sawtooth';
    osc.frequency.value = 440 * 2 ** ((note - 69) / 12);
    gain.gain.setValueAtTime(0.0001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.12 * velocity, context.currentTime + 0.015);
    osc.connect(gain);
    gain.connect(context.destination);
    osc.start();
    liveOscRef.current.set(note, {osc, gain});
    setActiveSynthNotes((current) => new Set(current).add(note));
  };

  const noteOff = (note: number) => {
    const context = getAudioContext();
    const node = liveOscRef.current.get(note);
    if (node) {
      node.gain.gain.cancelScheduledValues(context.currentTime);
      node.gain.gain.setTargetAtTime(0.0001, context.currentTime, 0.04);
      node.osc.stop(context.currentTime + 0.16);
      liveOscRef.current.delete(note);
    }
    setActiveSynthNotes((current) => {
      const next = new Set(current);
      next.delete(note);
      return next;
    });
  };

  const handleMidiMessage = (event: MIDIMessageEvent) => {
    if (!event.data) return;
    const [statusByte, note, rawVelocity] = Array.from(event.data);
    const command = statusByte & 0xf0;
    const velocity = rawVelocity / 127;
    const now = getAudioContext().currentTime;
    if (command === 0x90 && rawVelocity > 0) {
      noteOn(note, velocity);
      midiHeldRef.current.set(note, {velocity, start: now});
    } else if (command === 0x80 || command === 0x90) {
      noteOff(note);
      const held = midiHeldRef.current.get(note);
      const currentRecordState = recordStateRef.current;
      if (held && currentRecordState.mode === 'recording') {
        const loopStart = startedAtRef.current || now;
        const seconds = trackDuration(currentRecordState.trackId);
        const start = ((held.start - loopStart) % seconds + seconds) % seconds;
        midiNotesRef.current.push({
          note,
          velocity: held.velocity,
          start,
          duration: clamp(now - held.start, 0.05, seconds),
        });
      }
      midiHeldRef.current.delete(note);
    }
  };

  const recordMidi = async (trackId: string) => {
    const ready = midiReady || await ensureMidi();
    if (!ready) return;
    clearTimers();
    midiNotesRef.current = [];
    midiHeldRef.current.clear();
    const wait = countInMs();
    const startMs = performance.now() + wait;
    const seconds = trackDuration(trackId);
    const durationMs = seconds * 1000;
    setStatus('MIDI armed. Recording starts on the next bar edge.');
    updateCountdown('midi', trackId, startMs, durationMs);
    scheduleCountInClicks(wait);
    timersRef.current.push(window.setTimeout(() => {
      midiNotesRef.current = [];
    }, wait));
    timersRef.current.push(window.setTimeout(async () => {
      const notes = midiNotesRef.current;
      const buffer = await renderMidiNotes(notes, seconds);
      setTrackSourceBuffer(trackId, buffer, {kind: 'midi', notes});
      clearTimers();
      setRecordState({mode: 'idle'});
      setStatus(`Rendered ${notes.length} MIDI note(s) into a synth loop.`);
    }, wait + durationMs));
  };

  const transformWithMagenta = async (trackId: string) => {
    const track = tracksRef.current.find((candidate) => candidate.id === trackId);
    if (!track) return;
    const seconds = getTrackSeconds(track, bpm);
    const requestSeconds = seconds * (track.doubleGenerate ? 2 : 1) + (track.smoothFade ? 1 : 0);
    if (track.kind === 'audio' && track.sourceBuffer) {
      setStatus('Applying Stable Audio transform. Original capture stays available on the track.');
      const stableAudioPlan = getStableAudioGenerationPlan(seconds);
      const repeatedInput = await renderRepeatedAudioInput(track.sourceBuffer, stableAudioPlan);
      const audioBase64 = await blobToBase64(audioBufferToWavBlob(repeatedInput));
      const response = await fetch('/sa3/audio-to-audio', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          prompt: track.prompt,
          duration: stableAudioPlan.requestSeconds,
          initNoiseLevel: track.initNoiseLevel,
          steps: 8,
          audioBase64,
        }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        setStatus(error.hint || error.error || 'Stable Audio transform failed.');
        return;
      }
      const meta = response.headers.get('X-SA3-Meta');
      const decoded = await decodeBlob(await response.blob());
      const cropFrames = Math.round(stableAudioPlan.cropStart * decoded.sampleRate);
      const loop = await prepareGeneratedLoop(decoded, seconds, track.doubleGenerate, track.smoothFade, cropFrames);
      setTrackTransformedBuffer(trackId, loop);
      const generatedSeconds = meta ? JSON.parse(meta).duration.toFixed(1) : null;
      setStatus(generatedSeconds
        ? `Applied ${seconds.toFixed(1)}s Stable Audio transform from ${generatedSeconds}s generated audio after ${stableAudioPlan.cropStart.toFixed(1)}s warmup.`
        : `Applied ${seconds.toFixed(1)}s Stable Audio transform.`);
      return;
    }

    const midiNotes = track.kind === 'midi'
      ? repeatMidiNotes(track.notes, seconds, requestSeconds)
      : [];
    setStatus('Applying Magenta transform. Original capture stays available on the track.');
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: track.prompt,
        model,
        bpm,
        duration: requestSeconds,
        temperature: 1.25,
        topK: 40,
        cfgMusicCoCa: track.promptStrength,
        cfgNotes: track.notes.length ? 4.0 : 1.0,
        cfgDrums: 1.0,
        drums: track.drumless ? [0] : undefined,
        midiNotes,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      setStatus(error.hint || error.error || 'Magenta generation failed.');
      return;
    }
    const meta = response.headers.get('X-Magenta-Meta');
    const decoded = await decodeBlob(await response.blob());
    const loop = await prepareGeneratedLoop(decoded, seconds, track.doubleGenerate, track.smoothFade);
    setTrackTransformedBuffer(trackId, loop);
    const generatedSeconds = meta ? JSON.parse(meta).duration.toFixed(1) : null;
    setStatus(generatedSeconds
      ? `Applied ${seconds.toFixed(1)}s transform from ${generatedSeconds}s generated audio.`
      : `Applied ${seconds.toFixed(1)}s Magenta transform.`);
  };

  const addTrack = () => {
    updateTracks((current) => [...current, makeTrack(current.length)]);
  };

  const clearTrack = (trackId: string) => {
    updateTracks((current) => current.map((track) => track.id === trackId ? {...track, sourceBuffer: null, transformedBuffer: null, playbackMode: 'source', notes: [], kind: 'audio'} : track));
  };

  const changeTrackBars = async (trackId: string, bars: number) => {
    const track = tracksRef.current.find((candidate) => candidate.id === trackId);
    if (!track) return;
    const seconds = barsToSeconds(bars, bpm);
    const sourceBuffer = track.sourceBuffer ? await normalizeLoopBuffer(track.sourceBuffer, seconds) : null;
    const transformedBuffer = track.transformedBuffer ? await normalizeLoopBuffer(track.transformedBuffer, seconds) : null;
    updateTracks((current) =>
      current.map((item) =>
        item.id === trackId ? {...item, bars, sourceBuffer, transformedBuffer} : item,
      ),
    );
  };

  const keyboardNotes = ['A', 'W', 'S', 'E', 'D', 'F', 'T', 'G', 'Y', 'H', 'U', 'J'];

  return (
    <main className="app">
      <section className="transport-panel">
        <div className="brand-strip">
          <div>
            <p className="eyebrow">MAGENTA RT LOOP LAB</p>
            <h1>Tape Loop Console</h1>
          </div>
          <div className="reel-pair" aria-hidden="true">
            <div className="reel"><Disc3 size={42} /></div>
            <div className="reel"><Disc3 size={42} /></div>
          </div>
        </div>

        <div className="transport-row">
          <button className="transport-button primary" onClick={isPlaying ? stopPlayback : startPlayback} title={isPlaying ? 'Stop' : 'Play'}>
            {isPlaying ? <Square size={22} /> : <Play size={22} />}
          </button>
          <button className="transport-button" onClick={() => { stopPlayback(); startPlayback(); }} title="Restart">
            <RotateCcw size={20} />
          </button>
          <button
            className={`transport-button ${metronomeEnabled ? 'enabled' : ''}`}
            onClick={() => setMetronomeEnabled((value) => !value)}
            title={metronomeEnabled ? 'Metronome on' : 'Metronome off'}
            aria-label={metronomeEnabled ? 'Metronome on' : 'Metronome off'}
          >
            <span className="metro-mark">M</span>
          </button>
          <div className="loop-dial" style={{background: `conic-gradient(#ff5c39 ${progressDegrees}deg, #171717 0deg)`}}>
            <div>{visibleBeat}</div>
            <span>BEAT</span>
          </div>
          <label className="machine-field">
            <span>BPM</span>
            <input type="number" min="60" max="180" value={bpm} onChange={(event) => setBpm(Number(event.target.value))} />
          </label>
          <label className="machine-field wide">
            <span>MODEL</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {Array.from(new Set(['mrt2_small', 'mrt2_base', ...serverModels])).map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
        </div>

        <div className={`status-tape ${recordState.mode !== 'idle' ? 'hot' : ''}`}>
          <Radio size={16} />
          <span>
            {recordState.mode === 'idle'
              ? status
              : `${recordState.mode === 'count-in' ? 'COUNT-IN' : 'REC'} ${recordState.remaining.toFixed(1)}s`}
          </span>
        </div>
      </section>

      <section className="workspace">
        <div className="track-stack">
          <div className="section-heading">
            <h2>Loops</h2>
            <button className="icon-button" onClick={addTrack} title="Add track"><Plus size={18} /></button>
          </div>
          {tracks.map((track, index) => (
            <article
              key={track.id}
              className={`track-card ${selectedTrack?.id === track.id ? 'selected' : ''}`}
              style={{'--track-color': track.color} as React.CSSProperties}
              onClick={() => setSelectedTrackId(track.id)}
            >
              <div className="track-leds">
                <span className={recordState.mode !== 'idle' && recordState.trackId === track.id ? 'lit' : ''} />
                <span className={track.active ? 'lit green' : ''} />
              </div>
              <div className="track-main">
                <div className="track-title">
                  <strong>{track.name}</strong>
                  <small>
                    {track.kind.toUpperCase()} {track.bars} BAR {
                      track.sourceBuffer
                        ? (track.transformedBuffer ? `PLAY ${track.playbackMode === 'source' ? 'SRC' : 'AI'}` : 'SOURCE')
                        : (track.transformedBuffer ? 'AI ONLY' : 'EMPTY')
                    }
                  </small>
                </div>
                <input
                  value={track.prompt}
                  onChange={(event) => updateTracks((current) => current.map((item) => item.id === track.id ? {...item, prompt: event.target.value} : item), false)}
                  onClick={(event) => event.stopPropagation()}
                />
                <div className="capture-strip">
                  <CaptureGraphic track={track} bpm={bpm} />
                  <div
                    className="track-playhead"
                    style={{left: `${((playhead % getTrackSeconds(track, bpm)) / getTrackSeconds(track, bpm)) * 100}%`}}
                  />
                </div>
              </div>
              <div className="track-controls" onClick={(event) => event.stopPropagation()}>
                <button className="icon-button" onClick={() => updateTracks((current) => current.map((item) => item.id === track.id ? {...item, active: !item.active} : item))} title={track.active ? 'Pause loop' : 'Start loop'}>
                  {track.active ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <button className="icon-button red" onClick={() => recordAudio(track.id)} title="Record audio"><Mic size={16} /></button>
                <button className="icon-button yellow" onClick={() => recordMidi(track.id)} title="Record MIDI"><Piano size={16} /></button>
                <button className="icon-button blue" onClick={() => transformWithMagenta(track.id)} title="Apply Magenta transform"><Wand2 size={16} /></button>
                <button className="icon-button" onClick={() => clearTrack(track.id)} title="Clear"><Trash2 size={16} /></button>
                <label className="mini-select" title="Loop bars">
                  <span>BARS</span>
                  <select
                    value={track.bars}
                    onChange={(event) => void changeTrackBars(track.id, Number(event.target.value))}
                  >
                    {[1, 2, 4, 8, 16].map((value) => <option key={value} value={value}>{value}</option>)}
                  </select>
                </label>
                {track.sourceBuffer && track.transformedBuffer && (
                  <button
                    className={`mode-toggle ${track.playbackMode === 'source' ? 'source' : 'transformed'}`}
                    onClick={() => updateTracks((current) => current.map((item) => item.id === track.id ? {...item, playbackMode: item.playbackMode === 'source' ? 'transformed' : 'source'} : item))}
                    title={track.playbackMode === 'source' ? 'Hear transformed audio' : 'Hear original capture'}
                  >
                    {track.playbackMode === 'source' ? 'SRC' : 'AI'}
                  </button>
                )}
                <input
                  className="volume"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={track.volume}
                  onChange={(event) => updateTracks((current) => current.map((item) => item.id === track.id ? {...item, volume: Number(event.target.value)} : item), false)}
                  title="Volume"
                />
              </div>
              <div className="track-number">{String(index + 1).padStart(2, '0')}</div>
            </article>
          ))}
        </div>

        <aside className="control-bay">
          <div className="scope">
            {tracks.map((track, index) => (
              <div key={track.id} className="scope-row">
                <div className="scope-progress" style={{left: `${((playhead % getTrackSeconds(track, bpm)) / getTrackSeconds(track, bpm)) * 100}%`}} />
                <span style={{background: track.color, transform: `scaleX(${track.sourceBuffer ? 1 : 0.08})`}} />
                <i>{index + 1}</i>
              </div>
            ))}
          </div>

          <div className="patch-panel">
            <h2>{selectedTrack?.name ?? 'Loop'} Patch</h2>
            <div className="prompt-box">
              <Bot size={18} />
              <textarea
                value={selectedTrack?.prompt ?? ''}
                onChange={(event) => {
                  const value = event.target.value;
                  updateTracks((current) => current.map((track) => track.id === selectedTrack?.id ? {...track, prompt: value} : track), false);
                }}
              />
            </div>
            {selectedTrack && (
              <div className="detail-sliders">
                <label>
                  <span>Prompt strength</span>
                  <strong>{selectedTrack.promptStrength.toFixed(1)}</strong>
                  <input
                    type="range"
                    min="-1"
                    max="7"
                    step="0.1"
                    value={selectedTrack.promptStrength}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      updateTracks((current) => current.map((track) => track.id === selectedTrack.id ? {...track, promptStrength: value} : track), false);
                    }}
                  />
                </label>
                <label>
                  <span>Init noise</span>
                  <strong>{selectedTrack.initNoiseLevel.toFixed(2)}</strong>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={selectedTrack.initNoiseLevel}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      updateTracks((current) => current.map((track) => track.id === selectedTrack.id ? {...track, initNoiseLevel: value} : track), false);
                    }}
                  />
                </label>
                <div className="detail-toggles">
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedTrack.drumless}
                      onChange={(event) => {
                        const checked = event.target.checked;
                        updateTracks((current) => current.map((track) => track.id === selectedTrack.id ? {...track, drumless: checked} : track), false);
                      }}
                    />
                    <span>No drums</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedTrack.doubleGenerate}
                      onChange={(event) => {
                        const checked = event.target.checked;
                        updateTracks((current) => current.map((track) => track.id === selectedTrack.id ? {...track, doubleGenerate: checked} : track), false);
                      }}
                    />
                    <span>Double gen</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedTrack.smoothFade}
                      onChange={(event) => {
                        const checked = event.target.checked;
                        updateTracks((current) => current.map((track) => track.id === selectedTrack.id ? {...track, smoothFade: checked} : track), false);
                      }}
                    />
                    <span>Smooth fade</span>
                  </label>
                </div>
              </div>
            )}
            <div className="action-grid">
              <button onClick={() => selectedTrack && recordAudio(selectedTrack.id)}><Mic size={18} /> Audio</button>
              <button onClick={() => selectedTrack && recordMidi(selectedTrack.id)}><Piano size={18} /> MIDI</button>
              <button onClick={() => selectedTrack && transformWithMagenta(selectedTrack.id)}><Wand2 size={18} /> Transform</button>
            </div>
          </div>

          <div className="keys">
            <div className="section-heading">
              <h2>MIDI Monitor</h2>
              <button onClick={ensureMidi} className="small-button">{midiReady ? 'READY' : 'ENABLE'}</button>
            </div>
            <div className="keyboard">
              {keyboardNotes.map((key, index) => {
                const midi = 60 + index;
                return <span key={key} className={activeSynthNotes.has(midi) ? 'down' : ''}>{key}</span>;
              })}
            </div>
          </div>

          <div className="architecture">
            <h2>Signal Plan</h2>
            <p><Circle size={10} /> Browser transport, capture, MIDI and WebAudio loops.</p>
            <p><Circle size={10} /> Python backend for prompt-to-audio Magenta generation.</p>
            <p><Circle size={10} /> Later: swap backend for streaming C++ engine and audio-to-audio transforms.</p>
          </div>
        </aside>
      </section>
    </main>
  );
}

const container = document.getElementById('root')!;
const hotWindow = window as typeof window & {__magentaLoopRoot?: Root};
hotWindow.__magentaLoopRoot ??= createRoot(container);
hotWindow.__magentaLoopRoot.render(<App />);
