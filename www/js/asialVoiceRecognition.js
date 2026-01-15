
// ------------- グローバル変数 -------------
let model, labels;

// ------------- 音声の基本パラメータ -------------
// このサンプルは「16kHz / 1秒」の固定長クリップを前提にしています。
const SR = 16000;               // サンプリング周波数 (Hz)
const CLIP_SECONDS = 1.0;       // 1回の推論に使う長さ（秒）
const CLIP_SAMPLES = SR * CLIP_SECONDS; // 1秒ぶんのサンプル数（=16000）

// ------------- 特徴量（STFT / Mel / MFCC）のパラメータ -------------
// ★ここは学習時と一致している必要があります（窓長/ステップ/FFT長が違うと次元や分布がズレる）
const frameLength = 640; // 40ms (= 16000Hz * 0.04s)
const frameStep   = 320; // 20ms (= 16000Hz * 0.02s)
const fftLength   = 1024;

// Melフィルタバンクの数と、最終的に使うMFCC係数数
const numMelBins  = 40;
const numMfcc     = 13;

// Melフィルタの周波数レンジ
const lowerEdgeHz = 20.0;
const upperEdgeHz = SR / 2; // ナイキスト周波数（= 8000Hz）

// STFTの出力（rfft）の周波数bin数
// rfftのユニークbin数 = fftLength/2 + 1
const numSpectrogramBins = fftLength / 2 + 1;

// ------------- キャッシュ（毎フレーム作らない） -------------
// Mel重み行列とDCT行列は毎回作ると重いので、初回だけ作って保持します。
let melW = null;   // shape: [numSpectrogramBins, numMelBins]
let dctM = null;   // shape: [numMelBins, numMfcc]

// ------------- Hz <-> Mel 変換 -------------
function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// ------------- Melフィルタバンク重み行列を生成 -------------
// 目的：
//   |spectrogram| (T x numSpectrogramBins) に掛けることで
//   mel (T x numMelBins) を得るための重み行列を作る
//
// 返り値：
//   shape [numSpectrogramBins, numMelBins]
//   ※あとで spectrogram.matMul(melW) する想定
function buildMelWeightMatrix(numMelBins, numSpectrogramBins, sampleRate, lowerEdgeHz, upperEdgeHz) {
  // numSpectrogramBins = fftLength/2 + 1 なので、fftLength は (numSpectrogramBins - 1) * 2 で戻せる
  const fftLen = (numSpectrogramBins - 1) * 2;

  // 周波数レンジをMelスケールに変換
  const lowerMel = hzToMel(lowerEdgeHz);
  const upperMel = hzToMel(upperEdgeHz);

  // Mel上で等間隔な「三角フィルタ」の中心点（+両端）を作る
  // 三角フィルタは numMelBins 個必要なので、境界も含めて numMelBins+2 点
  const melPoints = new Array(numMelBins + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = lowerMel + (upperMel - lowerMel) * (i / (numMelBins + 1));
  }

  // Mel点をHzに戻す
  const hzPoints = melPoints.map(melToHz);

  // Hz -> FFT bin index へ変換
  // (fftLen+1) を掛けているのは実装上の一般的な換算（学習実装に合わせるのが大事）
  const bin = hzPoints.map(hz => Math.floor((fftLen + 1) * hz / sampleRate));

  // 重み行列本体（2Dを1Dに詰めて最後にtensor2d化）
  // shape: [numSpectrogramBins, numMelBins]
  const data = new Float32Array(numSpectrogramBins * numMelBins);

  // m番目の三角フィルタ（m=1..numMelBins）を作る
  for (let m = 1; m <= numMelBins; m++) {
    // 三角の左端/中心/右端（FFT bin）
    const f0 = bin[m - 1];
    const f1 = bin[m];
    const f2 = bin[m + 1];

    // 上り部分：f0 -> f1 で 0 -> 1
    for (let k = f0; k < f1; k++) {
      if (k >= 0 && k < numSpectrogramBins && f1 !== f0) {
        const w = (k - f0) / (f1 - f0);
        data[k * numMelBins + (m - 1)] = w;
      }
    }

    // 下り部分：f1 -> f2 で 1 -> 0
    for (let k = f1; k < f2; k++) {
      if (k >= 0 && k < numSpectrogramBins && f2 !== f1) {
        const w = (f2 - k) / (f2 - f1);
        data[k * numMelBins + (m - 1)] = Math.max(0, w);
      }
    }
  }

  return tf.tensor2d(data, [numSpectrogramBins, numMelBins], 'float32');
}

// ------------- DCT-II（直交正規化）行列を生成 -------------
// 目的：
//   log-mel (T x numMelBins) に掛けて MFCC (T x numMfcc) を得る
//
// 返り値：
//   shape [numMelBins, numMfcc]
function buildDctMatrix(numMelBins, numMfcc) {
  const data = new Float32Array(numMelBins * numMfcc);

  // k=0 とそれ以外でスケールが違う（直交正規化）
  const scale0 = 1 / Math.sqrt(numMelBins);
  const scale  = Math.sqrt(2 / numMelBins);

  // n: mel bin index, k: mfcc index
  for (let k = 0; k < numMfcc; k++) {
    for (let n = 0; n < numMelBins; n++) {
      const basis = Math.cos(Math.PI / numMelBins * (n + 0.5) * k);
      data[n * numMfcc + k] = (k === 0 ? scale0 : scale) * basis;
    }
  }
  return tf.tensor2d(data, [numMelBins, numMfcc], 'float32');
}

// ------------- 特徴量変換用行列の初期化（キャッシュ） -------------
function ensureFeatureMatrices() {
  if (!melW) melW = buildMelWeightMatrix(numMelBins, numSpectrogramBins, SR, lowerEdgeHz, upperEdgeHz);
  if (!dctM) dctM = buildDctMatrix(numMelBins, numMfcc);
}

// -----------------------------------------------------------------------------
// 音声クリップ（1秒, 16kHz想定）をモデル入力用のMFCCテンソルに変換する
//
// 入力:
//   audio1d: Float32Array などの1次元配列
//            長さは CLIP_SAMPLES (= SR * CLIP_SECONDS) を想定
//
// 出力:
//   tf.Tensor, shape: [1, T, numMfcc(=13), 1]
//     - 先頭の 1 : バッチ次元（1クリップ分）
//     - T        : 時間フレーム数（frameLength / frameStep / 入力長で決まる）
//     - 13       : MFCC 次元（各フレームの特徴量数）
//     - 最後の 1 : チャンネル次元（CNN等「画像扱い」の入力形状に合わせるため）
//
// 処理の流れ:
//   wav(波形) → STFT(複素スペクトル) → |.|(振幅スペクトログラム)
//          → Melフィルタバンク → log → DCT → MFCC
//
// メモリ:
//   tf.tidy() により、この関数内で生成した中間Tensorは自動的に破棄される
//   （return する MFCC テンソルだけが tidy の外に残る）
// -----------------------------------------------------------------------------
function audioToMfcc(audio1d) {
  // Mel重み行列(melW)とDCT行列(dctM)を未作成なら生成してキャッシュする
  ensureFeatureMatrices();

  return tf.tidy(() => {
    // 1) 波形（時間領域）: JS配列/TypedArray -> Tensor1D へ
    const wav = tf.tensor1d(audio1d);

    // 2) STFT: 短時間フーリエ変換（複素数）
    //    shape: [T, numSpectrogramBins]（numSpectrogramBins = fftLength/2 + 1）
    const stft = tf.signal.stft(wav, frameLength, frameStep, fftLength);

    // 3) 複素スペクトル -> スペクトログラムへ
    //    ここでは振幅（magnitude）= abs を採用
    //    ※学習時が「パワー（magnitude^2）」なら square() 等に合わせる必要あり
    const spectrogram = tf.abs(stft); // shape: [T, numSpectrogramBins]

    // 4) Melフィルタバンク適用: 周波数bin -> Mel帯域へ集約
    //    (T x bins) @ (bins x mel) => (T x mel)
    const mel = spectrogram.matMul(melW); // shape: [T, numMelBins]

    // 5) log-Mel: 強度を対数圧縮（0回避のため微小値を加算）
    const logMel = tf.log(mel.add(1e-6)); // shape: [T, numMelBins]

    // 6) DCTでMFCCへ: (T x mel) @ (mel x mfcc) => (T x mfcc)
    //    Mel帯域の情報を少数次元（ここでは13）に圧縮し、音声識別に使いやすくする
    const mfcc = logMel.matMul(dctM); // shape: [T, numMfcc]

    // 7) モデル入力形状に合わせて次元追加
    //    [T, 13] -> [T, 13, 1] -> [1, T, 13, 1]
    return mfcc.expandDims(-1).expandDims(0);
  });
}

// ------------- モデルとラベルの読み込み -------------
async function loadAssets() {
  // 例: ./tfjs_model_tfonly/model.json と ./tfjs_model_tfonly/labels.json がある想定
  // model.json は tfjs-converter などで出力されたKeras互換のモデル
  model = await tf.loadLayersModel("./tfjs_model/model.json");

  // labels.json は { "labels": ["silence", "yes", ...] } のような形式を想定
  labels = await (await fetch("./tfjs_model/labels.json")).json();

  // ラベル一覧を画面に表示
  document.querySelector("#words").textContent = labels.labels.join(", ");
}

// ------------- マイク入力をループして推論する -------------
async function startMicLoop(onaudioprocessCallback) {
  // マイク許可を取り、音声ストリームを取得
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // AudioContext を 16kHz 指定で生成（環境により厳密に16kHzにならない場合もあります）
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });

  // MediaStream -> AudioNode
  const source = audioCtx.createMediaStreamSource(stream);

  // ScriptProcessorNode（レガシーAPI）
  // 4096サンプル単位でコールバックが来る（16kHzなら約256msごと）
  // ※本格運用なら AudioWorklet が推奨ですが、最小構成としてScriptProcessorを使用
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  // ここに入力音声を溜めていく（可変長）
  let buffer = new Float32Array(0);
  let running = true;

  // 音声が来るたびに呼ばれる
  processor.onaudioprocess = async (e) => {
    // モノラル1ch想定で0番を取る
    const input = e.inputBuffer.getChannelData(0);

    // 既存bufferの末尾に input を連結
    const tmp = new Float32Array(buffer.length + input.length);
    tmp.set(buffer, 0);
    tmp.set(input, buffer.length);
    buffer = tmp;

    // 1秒分たまったら推論
    // （重い場合は、推論間隔を長くする/キュー化/非同期制御などを検討）
    if (buffer.length >= CLIP_SAMPLES) {
      // 先頭から1秒を切り出し
      const clip = buffer.slice(0, CLIP_SAMPLES);

      // 次回に向けて半分(0.5秒)だけ残して捨てる = 0.5秒オーバーラップ
      buffer = buffer.slice(Math.floor(CLIP_SAMPLES / 2));

      // 特徴量化してモデル入力へ
      const x = audioToMfcc(clip);

      // 推論
      // model.predict は通常 Tensor を返す（分類なら shape [1, numLabels] 等）
      const y = model.predict(x);

      // Tensor -> JS配列へ（GPU/wasm→CPUへ転送が発生）
      const scores = await y.data();

      // tidy外のTensorなので明示破棄
      x.dispose();
      y.dispose();

      if(onaudioprocessCallback && typeof onaudioprocessCallback === 'function') {
        onaudioprocessCallback(scores, labels);
      }
    }
  };

  // ノード接続
  // source -> processor へ入力を流し、
  // processor -> destination へ繋がないとScriptProcessorが動かない実装が多いです
  source.connect(processor);
  processor.connect(audioCtx.destination);

  // 停止用コントローラを返す
  return {
    stop: async () => {
      running = false;

      try { processor.disconnect(); } catch {}
      try { source.disconnect(); } catch {}

      try {
        stream.getTracks().forEach(t => t.stop());
      } catch {}

      try {
        await audioCtx.close();
      } catch {}
    }
  };
}
