"""Validation checks for observation processing.
"""
from datetime import datetime


def check_obsnme_suffix(obsnme_date_suffix, obsnme_suffix_format,
                        function_name, obsdata=None):
    format_type = 'date-based suffix'
    if not obsnme_date_suffix:
        format_type = 'stress period-based suffix'
    error_msg = (f"{function_name}: obsnme_suffix_format {obsnme_suffix_format} "
                  "doesn't appear to be compatible with "
                  f"obsnme_date_suffix={obsnme_date_suffix} ({format_type})")
    if obsnme_date_suffix:
        if '%' not in obsnme_suffix_format:
            raise ValueError(error_msg)
        if set(obsnme_suffix_format).intersection({'{', '}'}):
            raise ValueError(error_msg)
    else:
        if 'd' not in obsnme_suffix_format:
            raise ValueError(error_msg)
        try:
            f"{0:{obsnme_suffix_format.strip('{:}')}}"
        except:
            raise ValueError(error_msg)
    if obsdata is not None:
        prefixes, suffixes = zip(*obsdata['obsnme'].str.split('_'))
        for suffix in suffixes:
            if obsnme_date_suffix:
                try:
                    datetime.strptime(suffix, obsnme_suffix_format)
                except ValueError as e:
                    print(e)
                    print((f"observation suffix {suffix} "
                        "is incompatible with "
                        f"obsnme_suffix_format {obsnme_suffix_format}"
                        ))
            else:
                try:
                    int(suffix)
                except ValueError as e:
                    print(e)
                    print((f"observation suffix {suffix} "
                        "is incompatible with "
                        f"obsnme_suffix_format {obsnme_suffix_format}"
                        ))
