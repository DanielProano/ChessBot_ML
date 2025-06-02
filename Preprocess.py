import zstandard
import json
from io import TextIOWrapper

with open("Dataset/lichess_evas.zst", "rb") as in_file:
    zst = zstandard.ZstdDecompressor()
    with zst.stream_reader(in_file) as reader:
        with open("Dataset/cleaned_dataset", "w", encoding="utf-8") as out_file:
            with open("Dataset/test_dataset", "w", encoding="utf-8") as test_file:
                text = TextIOWrapper(reader)
                for line_num, line in enumerate(text):
                    try:
                        py_obj = json.loads(line)
                        evals = py_obj["evals"]
                        if evals is None:
                            continue
                        pvs = evals[0].get("pvs")
                        if pvs is None:
                            continue
                        cp = pvs[0].get("cp")
                        if cp is None:
                            continue
                        output = json.dumps({
                            "fen": py_obj["fen"],
                            "cp": cp
                        }) + "\n"
                        if line_num < 1000:
                            test_file.write(output)
                        else:
                            out_file.write(output)
                    except json.JSONDecodeError:
                        pass