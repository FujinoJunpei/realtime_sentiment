import pandas as pd


def get_sheet_as_df(service, sheet_id, sheet_name='シート1'):
    request = service.values().get(
        spreadsheetId=sheet_id, range=sheet_name)
    response = request.execute()
    table = response['values']
    headers = table.pop(0)
    result = pd.DataFrame(table, columns=headers)
    return result


def update_values_by_range(
        service, sheet_id, values,
        col_start, row_start, col_end, row_end,
        majorDimension=None, sheet_name='シート1'):
    update_range = (
        f'{sheet_name}!{col_start}{row_start}:{col_end}{row_end}')
    request_body = {'values': values}
    request = service.values().update(
        spreadsheetId=sheet_id, range=update_range, body=request_body,
        valueInputOption='USER_ENTERED')
    request.execute()
