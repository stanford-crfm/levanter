import tempfile

from levanter.data.sharded_datasource import AudioTextUrlDataSource, _sniff_format_for_dataset, ParquetDataSource
from test_utils import skip_if_no_soundlibs


def test_sniff_format_for_json():
    # this tests where some people use ".json" to mean a jsonlines file
    # and others use it to mean a json file

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'[{"text": "hello world"}, {"text": "hello world!"]')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{"text": "hello world"}\n{"text": "hello world!"}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".jsonl"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{\n"ids": [1, 2, 3]\n}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"


def test_sniff_format_for_parquet():

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        table = pa.table({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        pq.write_table(table, f.name)
        f.flush()

        assert _sniff_format_for_dataset(f.name) == ".parquet"
    

@skip_if_no_soundlibs
def test_resolve_audio_pointer():
    AudioTextUrlDataSource.resolve_audio_pointer("https://ccrma.stanford.edu/~jos/mp3/trumpet.mp3", 16_000)


def test_basic_parquet_datasource_read_row():

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        # Create a simple dataset
        data = {
            "column1": ["value1", "value2", "value3"],
            "column2": [10, 20, 30]
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, f.name)

        # Instantiate the ParquetDataSource
        datasource = ParquetDataSource([f.name])

        # sanity check: Read data starting from row 1
        row_data = list(datasource.open_shard_at_row(shard_name=f.name.replace(".", "_"), row=1))

        # Verify the output
        assert len(row_data) == 2  # We expect 2 rows starting from index 1
        assert row_data[0]["column1"] == "value2"
        assert row_data[0]["column2"] == 20
        assert row_data[1]["column1"] == "value3"
        assert row_data[1]["column2"] == 30