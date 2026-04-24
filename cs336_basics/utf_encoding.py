

test_string = "hello kai how are you"

utf8_encoded = test_string.encode('utf-8')

# print(list(utf8_encoded))

# print(list(test_string.encode('utf-16')))

# print(list(test_string.encode('utf-32')))

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# This won't run!
print(decode_utf8_bytes_to_str_wrong("héllo world!".encode("utf-8")))