import pathlib

folder_to_export = pathlib.Path(r'C:\Users\hx21262\Downloads\TEMP_XNAT\100_20200414\resources\RAW\files')

file_name = '66862.Anonymous.CT..601.RAW.20200414.110431.963978.2020.07.03.21.11.10.086000.1309236364.zip'

from upload_utils import PARAMIKO_PROTOCOL

paramiko_protocol = PARAMIKO_PROTOCOL(debug=False)

paramiko_protocol.sftp_transfer(
    str(folder_to_export.joinpath(file_name)),
    '/store/DAMTP/ev373/all_in_one'+'/test_zip.zip'
    )