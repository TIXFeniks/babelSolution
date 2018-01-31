from itertools import islice, chain
from batch_iterator import iterate_minibatches


def batch_generator(src_path, dst_path, batch_size=16, batches_per_epoch=None, skip_lines=0):
    with open(src_path) as f_src, open(dst_path) as f_dst:
        while True:
            num_lines = batches_per_epoch * batch_size if not batches_per_epoch is None else None

            f_src = islice(f_src, skip_lines, num_lines)
            f_dst = islice(f_dst, skip_lines, num_lines)

            batch = []

            for src_line, dst_line in zip(f_src, f_dst):
                if len(src_line) == 0 or len(dst_line) == 0: continue

                batch.append([src_line[:-1], dst_line[:-1]])

                if len(batch) >= batch_size:
                    yield (batch)
                    batch = []

            if batches_per_epoch is not None:
                raise StopIteration('File is read till the end, but too few batches were extracted')


def batch_generator_over_dataset(src, dst, batch_size=16, batches_per_epoch=None):
    for batch in iterate_minibatches(list(zip(src, dst)), batchsize=batch_size, shuffle=True):
        batch_src = [pair[0] for pair in batch]
        batch_dst = [pair[1] for pair in batch]

        yield (batch_src[0], batch_dst[0])
