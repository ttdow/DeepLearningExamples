import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('test3.jpg')

for detected in result:
    print(detected)