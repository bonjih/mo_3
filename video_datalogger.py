import os
import csv


class DataLogger:
    def trace(self, log_row: dict) -> None:
        pass


class CSVDataLogger(DataLogger):
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames

        if os.path.isfile(self.filename):
            os.rename(self.filename, self.filename + ".old")

        self._write_headers_if_needed()

    def _write_headers_if_needed(self):
        file_exists = os.path.exists(self.filename)
        if not file_exists:
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def trace(self, log_row):
        file_exists = os.path.exists(self.filename)
        if file_exists:
            with open(self.filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if not self._fieldnames_match(writer.fieldnames):
                    self.fieldnames = writer.fieldnames
                    self._write_headers_if_needed()
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if log_row is not None:
                    writer.writerow(log_row)

    def _fieldnames_match(self, new_fieldnames):
        return set(self.fieldnames) == set(new_fieldnames)
