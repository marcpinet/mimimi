import asyncio
import httpx
import zipfile
from pathlib import Path
from tqdm import tqdm as std_tqdm
from tqdm.asyncio import tqdm
import aiofiles


urls = [
    "https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1",
    "https://zenodo.org/records/3678171/files/dev_data_pump.zip?download=1",
    "https://zenodo.org/records/3678171/files/dev_data_slider.zip?download=1",
    "https://zenodo.org/records/3678171/files/dev_data_valve.zip?download=1",
]


async def download_file(url: str, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    file_path = dest_folder / filename
   
    if file_path.exists():
        print(f"{filename} already exists. Skipping download.")
        return file_path
   
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            async with aiofiles.open(file_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=filename) as progress:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
                        progress.update(len(chunk))
                        progress.update(len(chunk))
   
    print(f"Downloaded {filename}")
    return file_path

def extract_zip(zip_path: Path):
    extract_folder = zip_path.parent / zip_path.stem
   
    if extract_folder.exists() and any(extract_folder.iterdir()):
        print(f"Extraction folder {extract_folder.name} already exists and contains files. Skipping extraction.")
        return
   
    extract_folder.mkdir(parents=True, exist_ok=True)
   
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        
        with std_tqdm(total=total_files, desc=f"Extracting {zip_path.name}", unit="files") as extract_progress:
            for file in file_list:
                zip_ref.extract(file, extract_folder)
                extract_progress.update(1)
   
    print(f"Extraction of {zip_path.name} complete")

async def download_and_extract(url: str, download_folder: Path):
    file_path = await download_file(url, download_folder)
    await asyncio.to_thread(extract_zip, file_path)


async def main():
    download_folder = Path(__file__).resolve().parent
   
    tasks = [download_and_extract(url, download_folder) for url in urls]
    await asyncio.gather(*tasks)
    print("All files downloaded and extracted successfully!")


if __name__ == "__main__":
    asyncio.run(main())
