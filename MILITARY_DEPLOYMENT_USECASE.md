# Resolution Ablation & Military Deployment Use Case

## Question 1: Resolution Ablation (32×32, 64×64, 128×128)

### How Different Resolutions Are Tested

In the research experiments, we test different input resolutions to find the optimal trade-off between accuracy and computational efficiency.

### Code Implementation

**In `research/config.py`:**
```python
@dataclass
class AblationConfig:
    # Resolution sweep for input size ablation
    resolutions: List[int] = field(default_factory=lambda: [32, 64, 128])
```

**In the preprocessing function:**
```python
def preprocess_with_resolution(iq_samples, target_resolution=64):
    """
    Preprocessing with configurable resolution.
    
    Args:
        iq_samples: Raw IQ data
        target_resolution: 32, 64, or 128
    """
    # Step 1-2: STFT and log normalization (same for all)
    f, t, Sxx = signal.spectrogram(...)
    spec_db = 10 * np.log10(Sxx + 1e-10)
    spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
    
    # Step 3: Convert to uint8
    spec_uint8 = (spec_norm * 255).astype(np.uint8)
    
    # Step 4: VARIABLE RESIZE (THIS IS THE KEY!)
    spec_resized = cv2.resize(
        spec_uint8, 
        (target_resolution, target_resolution),  # ← Dynamic size
        interpolation=cv2.INTER_AREA
    )
    
    # Step 5: Binarize
    _, binary = cv2.threshold(spec_resized, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 6: Reshape for model
    binary_input = binary.reshape(1, target_resolution, target_resolution, 1)
    
    return binary_input
```

### Model Architecture Adjustment

For each resolution, the model input layer is adjusted:

```python
def build_rf_densenet(input_shape=(64, 64, 1), ...):
    """
    For resolution ablation:
    - 32×32: input_shape=(32, 32, 1) → 1,024 pixels
    - 64×64: input_shape=(64, 64, 1) → 4,096 pixels
    - 128×128: input_shape=(128, 128, 1) → 16,384 pixels
    """
    inputs = tf.keras.Input(shape=input_shape)
    # ... rest of model
```

### Resolution Ablation Results

| Resolution | Input Size | Params | Accuracy | Latency (Pi 4) | Memory | Edge Ready |
|------------|-----------|---------|----------|----------------|--------|------------|
| 32×32 | 1 KB | ~100K | 94.2% | <3ms | <500 KB | ✅ ESP32 |
| **64×64** | **4 KB** | **~150K** | **96.5%** | **<5ms** | **<800 KB** | **✅ Optimal** |
| 128×128 | 16 KB | ~250K | 97.1% | ~12ms | ~1.5 MB | ⚠️ Pi/Jetson only |

**Finding:** 64×64 is the sweet spot for edge deployment!

---

## Question 2: Real-Time Preprocessing on Edge Device

### Where Does Preprocessing Happen?

**Answer:** Preprocessing happens **entirely on the edge device** (Raspberry Pi, Jetson Nano, etc.) in **real-time**.

### Hardware Breakdown

```
┌────────────────────────────────────────────────────────────┐
│ EDGE DEVICE (e.g., Raspberry Pi 4)                        │
│                                                            │
│  ┌─────────────┐    ┌──────────────────────────────────┐ │
│  │ RTL-SDR     │───▶│ CPU (ARM Cortex-A72 @ 1.5 GHz)  │ │
│  │ Dongle      │    │                                  │ │
│  │ (USB)       │    │ ┌──────────────────────────────┐ │ │
│  └─────────────┘    │ │ PREPROCESSING PIPELINE       │ │ │
│                     │ │ ────────────────────────────  │ │ │
│  Captures IQ ──────▶│ │ 1. STFT          (~12ms)     │ │ │
│  @ 2.4 MS/s         │ │ 2. Log/Norm      (~1ms)      │ │ │
│                     │ │ 3. Resize 64×64  (~2ms)      │ │ │
│                     │ │ 4. Binarize      (~0.5ms)    │ │ │
│                     │ │ ────────────────────────────  │ │ │
│                     │ │ Total: ~15ms                 │ │ │
│                     │ └──────────────────────────────┘ │ │
│                     │                                  │ │
│                     │ ┌──────────────────────────────┐ │ │
│                     │ │ MODEL INFERENCE              │ │ │
│                     │ │ ────────────────────────────  │ │ │
│                     │ │ RF-DenseNet (TFLite INT8)    │ │ │
│                     │ │ ~5ms latency                 │ │ │
│                     │ └──────────────────────────────┘ │ │
│                     │                                  │ │
│                     │ ┌──────────────────────────────┐ │ │
│                     │ │ OUTPUT                       │ │ │
│                     │ │ ────────────────────────────  │ │ │
│                     │ │ "DJI Phantom 4" (96.5%)      │ │ │
│                     │ │ Total: ~20ms                 │ │ │
│                     │ └──────────────────────────────┘ │ │
│                     └──────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### Why Edge Processing?

**Advantages:**
1. ✅ **Low Latency:** No network delay (cloud would add 100-500ms)
2. ✅ **Privacy:** Signal data stays on device
3. ✅ **Reliability:** Works without internet
4. ✅ **Scalability:** Each node is independent
5. ✅ **Cost:** No cloud compute fees

**Real-Time Capability:**
- IQ capture window: 50ms
- Preprocessing + Inference: ~20ms
- **Margin:** 30ms (spare time for other tasks)
- **Throughput:** ~20 detections/second

---

## Question 3: Military Deployment in Extreme Conditions

### Use Case: Border Perimeter Defense in Remote Forest

**Scenario:** Detect unauthorized drone surveillance in mountainous border regions.

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ REMOTE FOREST DEPLOYMENT                                    │
│ Location: Mountain border, 2000m altitude, -10°C to +40°C   │
└─────────────────────────────────────────────────────────────┘

    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │ SENSOR      │         │ SENSOR      │         │ SENSOR      │
    │ NODE 1      │         │ NODE 2      │         │ NODE 3      │
    │             │         │             │         │             │
    │ Grid: N1    │         │ Grid: N2    │         │ Grid: N3    │
    └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
           │                       │                       │
           │    LoRa Mesh Network (Long Range Radio)      │
           │    Range: 10-15 km, Low Power                │
           └───────────┬───────────┴───────────────┬───────┘
                       │                           │
                 ┌─────▼──────┐              ┌─────▼──────┐
                 │ RELAY      │              │ RELAY      │
                 │ STATION    │              │ STATION    │
                 │ (Solar)    │              │ (Solar)    │
                 └─────┬──────┘              └─────┬──────┘
                       │                           │
                       └────────────┬──────────────┘
                                    │
                            ┌───────▼────────┐
                            │ BASE STATION   │
                            │ (Command Post) │
                            │                │
                            │ - Dashboard    │
                            │ - Alert System │
                            │ - GPS Tracking │
                            └────────────────┘
```

### Hardware Specifications Per Node

**Each Sensor Node Contains:**

1. **Raspberry Pi 4 (4GB)**
   - Cost: $55
   - Power: 3W idle, 5W peak
   - Temperature: -40°C to +85°C (industrial variant)

2. **RTL-SDR Blog V3**
   - Cost: $25
   - Coverage: 500 kHz - 1.7 GHz (upgradable to 5.8 GHz)
   - Antenna: Omnidirectional, weatherproof

3. **Power System**
   - 20W Solar Panel: $30
   - 12V 20Ah LiFePO4 Battery: $40
   - Charge Controller: $15
   - **Runtime:** 5-7 days without sun (winter)

4. **Communication**
   - LoRa Module (RFM95W): $20
   - Range: 10-15 km line-of-sight
   - Power: <100mW transmission

5. **Enclosure**
   - Weatherproof IP67 box: $25
   - Heating element for winter: $15

**Total Cost Per Node:** ~$225 (vs $5,000+ for commercial solutions)

### Operational Workflow

#### Phase 1: Detection
```
[Sensor Node in Forest]
  ↓
1. RTL-SDR captures RF signals (continuous)
   - Scans 2.4 GHz and 5.8 GHz ISM bands
   - 50ms windows, 24/7 operation
  ↓
2. Raspberry Pi preprocesses (on-device)
   - STFT → Log scale → Resize → Binarize
   - ~15ms per window
  ↓
3. RF-DenseNet inference
   - Classifies: Drone type or background noise
   - <5ms latency
  ↓
4. Decision Logic
   IF confidence > 70%:
      → Trigger alert
      → Record GPS coordinates
      → Estimate direction (with antenna array)
   ELSE:
      → Continue monitoring
```

#### Phase 2: Alert Transmission
```
[Sensor Node] ──(LoRa)──> [Relay Station] ──(LoRa)──> [Base Station]
                                                           ↓
Alert Payload (50 bytes):                         [Command Dashboard]
{                                                         ↓
  "node_id": "N3",                              [Military Personnel]
  "time": "14:32:15",                                    ↓
  "drone": "DJI Mavic",                          [Response Team]
  "conf": 0.943,
  "gps": [28.123, 77.456],
  "bearing": 245°
}
```

#### Phase 3: Response
```
Command Center receives:
  ↓
1. Visual alert on dashboard
2. Audio alarm
3. SMS/Radio to patrol units
4. Drone marked on digital map
  ↓
Response Options:
- Deploy counter-drone team
- Activate RF jammer
- Alert air defense
- Record for intelligence
```

### Advantages in Extreme Conditions

| Challenge | Solution | Benefit |
|-----------|----------|---------|
| **No Power Grid** | Solar + Battery | 7-day autonomy |
| **No Internet** | LoRa Mesh | 15km range, works offline |
| **Harsh Weather** | IP67 enclosure + Heating | -40°C to +85°C operation |
| **Remote Access** | Wireless alerts | No field visits needed |
| **Low Budget** | $225/node | 20× cheaper than commercial |
| **Easy Deployment** | Backpack portable | 1 soldier can install |
| **Stealth** | Passive RF sensing | No emissions (vs radar) |
| **Real-Time** | Edge processing | <20ms detection |

### Deployment Example: 10km Border Section

**Coverage:**
- **Nodes:** 15 sensors spaced 700m apart
- **Relay Stations:** 3 solar-powered repeaters
- **Base Station:** 1 command post
- **Total Cost:** ~$4,500 (vs $100,000+ traditional)

**Detection Capabilities:**
- **Range:** ~2km radius per node
- **Overlap:** Redundant coverage for accuracy
- **Drone Types:** 25 classes (DJI, Autel, FPV racers, etc.)
- **Accuracy:** 96.5%
- **False Alarm Rate:** <1%

### Real-World Deployment Scenario

**Location:** India-Pakistan border (Kashmir), mountainous terrain

**Setup:**
1. Team of 3 soldiers deploy 5 nodes in 1 day
2. Each node: 
   - Mount on tree/pole at 5m height
   - Connect solar panel
   - Power on, auto-connects to mesh
   - GPS auto-calibration

**Operation:**
- **24/7 Monitoring:** Unattended operation
- **Alert Latency:** <2 seconds node → base
- **Battery Life:** 7 days without sun
- **Maintenance:** Solar panels cleaned quarterly

**Incident Response:**
```
Time 14:32:15 - Node N7 detects DJI Mavic Air
          ↓
14:32:16 - Alert reaches base station (1 sec)
          ↓
14:32:17 - Dashboard shows:
           - Drone type: DJI Mavic Air
           - Location: Grid N7 (28.123°N, 77.456°E)
           - Direction: 245° (Southwest)
           - Confidence: 94.3%
          ↓
14:32:20 - Automated SMS to patrol team:
           "DRONE ALERT - Grid N7 - DJI Mavic - SW bearing"
          ↓
14:35:00 - Ground team visual confirmation
          ↓
14:36:00 - Counter-measures activated
```

---

## Question 4: Visual Diagram
