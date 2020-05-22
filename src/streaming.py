import yaml
from realtime_sentiment.lib.converters import save_dir
from realtime_sentiment.lib.auth import google_spreadsheet_auth
from realtime_sentiment.lib.spread_sheet import (
    get_sheet_as_df, update_values_by_range)


def main():
    service = google_spreadsheet_auth()
    target_jsonl = get_new_comments(service)
    with open(f"{save_dir('output')}/new_comments.jsonl", mode='w') as j:
        j.write(target_jsonl)


def get_new_comments(
        service,
        config_path='realtime_sentiment/config.yml',
        sheet_name='シート1'):
    with open(config_path, 'r', encoding='UTF-8') as yml:
        config = yaml.safe_load(yml)
    sheet_id = config['sheet_id']
    target_sheet = get_sheet_as_df(service, sheet_id, sheet_name)
    target_records = target_sheet[
        target_sheet.is_taken.astype(str, errors='ignore') != '1']
    if len(target_records) == 0:
        return ''
    else:
        target_ids = target_records.id.astype(int)
        mark_taken_records(service, target_ids, sheet_id, sheet_name)
        target_jsonl = target_records[['id', 'text']].to_json(
            orient='records', force_ascii=False, lines=True)
        return target_jsonl


def mark_taken_records(
        service, target_ids, sheet_id, sheet_name='シート1'):
    range_start = min(target_ids) + 2
    range_end = max(target_ids) + 2
    values = [[1]] * (range_end - range_start + 1)
    update_values_by_range(
        service, sheet_id, values,
        'D', range_start, 'D', range_end)


if __name__ == '__main__':
    main()
