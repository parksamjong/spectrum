import sounddevice as sd
devs = sd.query_devices()
inputs = [(i, d) for i, d in enumerate(devs) if d['max_input_channels'] > 0]
print(f"sounddevice ok - {len(inputs)} input device(s)")
for i, d in inputs:
    print(f"  [{i}] {d['name']}  ch={d['max_input_channels']}  sr={int(d['default_samplerate'])}")
