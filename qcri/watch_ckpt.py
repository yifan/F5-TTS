import os
import time
import glob
import argparse
import io
import matplotlib.pyplot as plt
import soundfile as sf

from azure.storage.blob import BlobServiceClient
from label_studio import APIClient
from f5_tts.api import F5TTS


def upload_file(args, blob_name, data):
    blob_service_client = BlobServiceClient(account_url=args.azure_storage_url, credential=args.azure_storage_token)
    container_client = blob_service_client.get_container_client(args.azure_storage_container)

    container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    return f'{args.azure_storage_url}/{args.azure_storage_container}/{blob_name}'


def infer(ckpt_file, vocab_file, ref_file, ref_text, gen_text):
    f5tts = F5TTS(
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
    )

    wav, sr, spec = f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=gen_text,
    )

    # Convert wav to raw bytes
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, wav, sr, format='WAV')
    wav_bytes.seek(0)

    # Convert spec to raw bytes (as an image)
    spec_bytes = io.BytesIO()
    plt.imsave(spec_bytes, spec, format='png')
    spec_bytes.seek(0)

    file_wave = os.path.join("tts", os.path.basename(ckpt_file).replace(".pt", ".wav"))
    file_spec = os.path.join("tts", os.path.basename(ckpt_file).replace(".pt", ".png"))

    return {
        "audio": upload_file(args, file_wave, data=wav_bytes.getvalue()),
        "spec": upload_file(args, file_spec, data=spec_bytes.getvalue()),
        "transcript": gen_text,
        "ckpt": ckpt_file,
    }


def create_task(args, ckpt_file):
    ref_file = args.ref_file
    ref_text = args.ref_text
    gen_text = args.gen_text
    vocab_file = args.vocab_file

    data = infer(ckpt_file, vocab_file, ref_file, ref_text, gen_text)
    client = APIClient(args.host, args.api_key, debug=args.debug)
    task = client.create_task(project_id=args.project_id, data=data)
    print(task)


def watch_directory(args):
    # Dictionary to store matched files
    file_dict = {
        f: True
        for f in glob.glob(args.path)
    }

    while True:
        # List all files in the directory matching the pattern using glob
        current_files = glob.glob(args.path)

        # Check for new files
        for f in current_files:
            if f not in file_dict:
                # generate sample and upload to label studio
                create_task(args, f)
                print(f)
                file_dict[f] = True

        # Wait for 1 minute
        time.sleep(60)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Watch a directory for new files matching a pattern.")
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--host', type=str, help='host of the Label Studio instance', default=os.environ.get('LABEL_STUDIO_URL', 'https://labelstudio.qcri.org'))
    parser.add_argument('--api-key', type=str, help='API key for the source workspace', default=os.environ.get('LABEL_STUDIO_API_KEY'))
    parser.add_argument('--azure-storage-url', type=str, help='azure storage url for files', default='https://qcristore.blob.core.windows.net')
    parser.add_argument('--azure-storage-token', type=str, help='azure storage sas token for files', default=os.environ.get('AZURE_STORAGE_TOKEN'))
    parser.add_argument('--azure-storage-container', type=str, help='azure storage sas token for files', default='labelstudio-p1')
    parser.add_argument('--vocab-file', type=str, help='vocab.txt file')
    parser.add_argument('--ref-file', type=str, help='reference audio file')
    parser.add_argument('--ref-text', type=str, help='reference text')
    parser.add_argument('--gen-text', type=str, help='generated text')
    parser.add_argument('--project-id', type=str, help='label studio project id')
    parser.add_argument("path", type=str, help="The directory to watch.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    watch_directory(args)