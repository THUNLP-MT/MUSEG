import multiprocessing, tqdm, os

SRC_DIR = './videos/et_bench'
TGT_DIR = './videos/et_bench_processed'


def proc(blob):
    os.makedirs(os.path.dirname(blob[1]), exist_ok=True)
    os.system(f'ffmpeg -y -i {blob[0]} -map 0:v {blob[1]} > /dev/null 2>&1')


if __name__ == '__main__':
    blobs = list()
    for path, _, files in os.walk(SRC_DIR):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.mkv') or file.endswith('.webp'):
                src = os.path.join(path, file)
                tgt = os.path.join(TGT_DIR, os.path.relpath(src, SRC_DIR))
                blobs.append((src, tgt))
            else:
                assert False

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap(proc, blobs), total=len(blobs)))
