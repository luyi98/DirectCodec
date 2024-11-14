import dac
from audiotools import AudioSignal

import os
import torch
import torchaudio


input_folder = "/data3/luyi/icassp2025/test_set/"
output_folder = "/data3/luyi/icassp2025/BSQ-20/"
model_path = "/data3/luyi/mycode/DAC-BSQ/runs/baseline-20/best/dac/weights.pth"

# Download a model
model = dac.utils.load_model(load_path = model_path)
model.to('cuda')

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))

            # Create output directory if not exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Load audio signal file
                signal = AudioSignal(input_path)

                # Encode audio signal as one long file
                # (may run out of GPU memory on long files)
                signal.to(model.device)

                with torch.no_grad():
                    x = model.preprocess(signal.audio_data, signal.sample_rate)
                    z, codes, _ = model.encode(x)

                    # Decode audio signal
                    y = model.decode(z)

                # # Alternatively, use the `compress` and `decompress` functions
                # # to compress long files.

                # signal = signal.cpu()
                # x = model.compress(signal)

                # # Save and load to and from disk
                # x.save("compressed.dac")
                # x = dac.DACFile.load("compressed.dac")

                # # Decompress it back to an AudioSignal
                # y = model.decompress(x)

                # Write to file
                # y.write(output_path)
                torchaudio.save(output_path, y.cpu().squeeze(0), model.sample_rate)
                print(output_path + " has been saved!")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")