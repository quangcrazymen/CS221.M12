from vncorenlp import VnCoreNLP
client = VnCoreNLP(address='http://127.0.0.1', port=8000)

print(client.pos_tag("Nhiều thi thể là binh sĩ và quan chức trong chính quyền cũ bị Taliban xử tử ."))