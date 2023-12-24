import string
import random
import math
import easyocr #imp --- img to text
import cv2  #imp  --- video capture
import qrcode #imp
# import numpy 
# from matplotlib import pyplot as plt

from ultralytics import YOLO  #imp

model = YOLO("license_plate_detector.pt")

classNames = ['AP02BP2454', 'AP28N3107', 'AS23M1264', 'ASM1634', 'CG10S0650', 'CH01AN0001', 'CRETA', 'DL10CG4693', 'DL12C3536', 'DL12CG6648', 'DL13S0155', 'DL14CTC0153', 'DL1N4268', 'DL3CAY2231', 'DL3CAY9324', 'DL3CCQ0107', 'DL6CAB123X', 'DL6CM6683', 'DL758Y1790', 'DL7CN5617', 'DL8CN6308', 'DL8CX4850', 'DUSTER', 'Devanagri', 'GJ01MW7581', 'GJ03JL0126', 'GJ051D9443', 'GJ05JD9759', 'GJ05JH2501', 'GJ07BR1336', 'GJ12FA0735', 'GJ15CD0564', 'GJ15D0564', 'GJ17TC214', 'GJ1KF1111', 'GJ7BB766', 'GJW115A1138', 'HD02NP0', 'HP896786', 'HPSX4000', 'HR11F7575', 'HR26AZ5927', 'HR26BA8008', 'HR26BC5514', 'HR26BP3543', 'HR26BR9044', 'HR26BU0375', 'HR26BU0380', 'HR26CB1900', 'HR26CE1485', 'HR26CF5868', 'HR26CH3604', 'HR26CK8571', 'HR26CM6005', 'HR26CP3135', 'HR26CR3302', 'HR26CT4063', 'HR26CT6702', 'HR26CU6799', 'HR26DA0471', 'HR26DA2330', 'HR26DA5443', 'HR26DG6167', 'HR26DK0830', 'HR26DK6475', 'HR26TC5656', 'HR26TC7099', 'HR26TC7303', 'HR26U7501', 'HR51AY9099', 'HR696969', 'HR99BJ6662', 'HR99EX6037', 'HR99HA4575', 'KA01D1330', 'KA01MA6989', 'KA031351', 'KA03AB3380', 'KA03MG2784', 'KA03MX5058', 'KA03NA8385', 'KA04ME9869', 'KA04MN3622', 'KA05HS4495', 'KA05MG1909', 'KA05MQ92', 'KA18P2987', 'KA19TR02', 'KA21M5519', 'KA29Z999', 'KA42TC131011', 'KA5122727', 'KA51AA3469', 'KA51MJ2143', 'KA51MJ8156', 'KL01AU585', 'KL01AX8000', 'KL01CA2555', 'KL01CC50', 'KL01KL0KL01', 'KL01KLKL01', 'KL02AF6363', 'KL05AK3300', 'KL06H5834', 'KL07BF2007', 'KL07BF5000', 'KL07CB8599', 'KL09AL9405', 'KL10AV6342', 'KL10AV6633', 'KL10AW2111', 'KL10AW2814', 'KL12G7531', 'KL13AA6340', 'KL16J3636', 'KL20K7561', 'KL25B2001', 'KL38F5008', 'KL43B2344', 'KL454455', 'KL49H5270', 'KL53E964', 'KL54A2670', 'KL54H369', 'KL55R2473', 'KL57A111', 'KL60N5344', 'KL63C8800', 'KL65H4383', 'KL7BZ99', 'ME02EE4077', 'MH010PO323', 'MH01AE8017', 'MH01AH8495', 'MH01AL9693', 'MH01AM5810', 'MH01AM7014', 'MH01AM9839', 'MH01AR5274', 'MH01AU0059', 'MH01AV7461', 'MH01AV8814', 'MH01AV8866', 'MH01AX1113', 'MH01AX3070', 'MH01BD1897', 'MH01BD2186', 'MH01BG7161', 'MH01BM4824', 'MH01BT0050', 'MH01BT4302', 'MH01BT9066', 'MH01BU1852', 'MH01BU5207', 'MH01BY7658', 'MH01CP2655', 'MH01CR7388', 'MH01CT1710', 'MH01CT9150', 'MH01CV6333', 'MH01CY6333', 'MH01DB1477', 'MH01DB2561', 'MH01DE2780', 'MH01DK4867', 'MH01DP5662', 'MH01DT1917', 'MH01JA8939', 'MH02AJ344', 'MH02AK5481', 'MH02AQ2299', 'MH02B16890', 'MH02BG9542', 'MH02BJ2456', 'MH02BK9793', 'MH02BM5048', 'MH02BQ9628', 'MH02BR4503', 'MH02BT6482', 'MH02CB4545', 'MH02CD3654', 'MH02CD7733', 'MH02CL4017', 'MH02CR4364', 'MH02CR8276', 'MH02CT2727', 'MH02CW8226', 'MH02DJ8952', 'MH02DN6980', 'MH02DN8718', 'MH02DQ0196', 'MH02DS3099', 'MH02DS4676', 'MH02DS9365', 'MH02DW8021', 'MH02DZ9898', 'MH02EE6842', 'MH02EE8407', 'MH02EH1077', 'MH02EK0837', 'MH02EK4399', 'MH02EK5817', 'MH02EN6828', 'MH02EP4454', 'MH02EP5740', 'MH02ER9194', 'MH02EU4077', 'MH02EU4884', 'MH02EZ2599', 'MH02EZ6482', 'MH02FD6534', 'MH02FN2783', 'MH02JA9655', 'MH02MA5324', 'MH02WA8344', 'MH038C6247', 'MH03AF1911', 'MH03AM3025', 'MH03AR5549', 'MH03AR7140', 'MH03AT3856', 'MH03BS4060', 'MH03BS7778', 'MH03BS8519', 'MH03CB0229', 'MH03CB6467', 'MH03CP1328', 'MH03CS6266', 'MH03DA4505', 'MH03DK2961', 'MH03DU5196', 'MH03DV2010', 'MH04CT3566', 'MH04DW8351', 'MH04DW9020', 'MH04EX7505', 'MH04GM7982', 'MH04HM4154', 'MH04HP6100', 'MH04JL5547', 'MH04JM3383', 'MH04JM8262', 'MH05DK1018', 'MH05DS8679', 'MH05EJ3005', 'MH06AW8929', 'MH06AZ3571', 'MH12BG7237', 'MH12DE1433', 'MH12DG7144', 'MH12FT9458', 'MH12FU1014', 'MH12JC2813', 'MH12NE8922', 'MH12RT1105', 'MH12SF3212', 'MH13BN4348', 'MH14DT8831', 'MH14DX9937', 'MH14DX9938', 'MH14EH5819', 'MH14EH7958', 'MH14EP4660', 'MH14EU3498', 'MH14EY5972', 'MH14FM6930', 'MH14FS5229', 'MH14GN9239', 'MH14TC206AN', 'MH14TC947', 'MH14TCD204', 'MH14TCE4', 'MH14TCF300', 'MH14TCF459', 'MH14TCF460', 'MH14TCP237', 'MH15BD8877', 'MH15TC554', 'MH20BN3525', 'MH20BQ20', 'MH20BY3665', 'MH20BY4465', 'MH20CS1938', 'MH20CS1941', 'MH20CS4946', 'MH20CS9817', 'MH20DJ0419', 'MH20DV2362', 'MH20EE045', 'MH20EE0943', 'MH20EE7597', 'MH20EE7598', 'MH20EE7601', 'MH20EJ0364', 'MH20EU9991', 'MH20TC189B', 'MH20TC640B', 'MH20TC830C', 'MH21V9926', 'MH43AF5037', 'MH43AL8464', 'MH43AR5466', 'MH43BK9513', 'MH43BP8173', 'MH43BU2401', 'MH46AD5258', 'MH46BF2342', 'MH46BV0688', 'MH46X9996', 'MH46Z8892', 'MH4705851', 'MH47AG4326', 'MH47AU1306', 'MH47AV6753', 'MH47AZ8323', 'MH47N2829', 'MH47N4570', 'MH47Y0205', 'MH47Y1124', 'MH48AW0091', 'MHD1CV9311', 'MP09CC1667', 'MP09CP9052', 'PB03AGT8979', 'PB08CX2959', 'PB11AG0774', 'PY01BL1155', 'RJ02CC0784', 'RJ09C84382', 'RJ27TA1143', 'RJ27TC0530', 'TERRANO', 'TN02BL9', 'TN02TC0143', 'TN07BU5427', 'TN07BV5200', 'TN09CH7770', 'TN19H3322', 'TN19S4523', 'TN19TC91', 'TN19TC94', 'TN21AT0479', 'TN21AT0480', 'TN21AT0492', 'TN21AU1153', 'TN21AU7234', 'TN21BC6225', 'TN21BY0166', 'TN21BZ0768', 'TN21TC31', 'TN21TC611', 'TN28BA9999', 'TN37CR4019', 'TN38BW1139', 'TN38BY4191', 'TN42R2697', 'TN43J0158', 'TN45BA1065', 'TN52U1580', 'TN58AM1', 'TN59AQ1515', 'TN59BE0939', 'TN66T5836', 'TN66U8215', 'TN74AH1413', 'TN74AL5074', 'TN74F3339', 'TN99F2378', 'TS08ER1643', 'TS09EB1458', 'UK07BA7252', 'UP16AB3726', 'UP16CT2233', 'UP16TC1366', 'UP32FH2653', 'UP50AS4535', 'UP65CM21149', 'UP80CR00', 'W0BNP300', 'WB06F9209', 'WB07D5106', 'WB74X7605', 'blur']

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    results = model.predict(source=frame, stream=True)
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            
            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
            
    
    # frame = cv2.resize(frame, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    
    if cv2.waitKey(20) % 0xFF == ord("d"):
        break

reader = easyocr.Reader(['en'])
result = reader.readtext(frame,paragraph=False)

print(result[-1][-2])
text = result[-1][-2]
cap.release()
cv2.destroyAllWindows() 

size = 5
Database = "DL 7C0 1939"  #represents origional database
identity = False


#Resident owner side:

def qrcheck():
    print("owner side:\n")
    gen = input("You want to generate QR code: ")
    if(gen == "yes"):
        rand_text = ''.join(random.choices(string.ascii_uppercase +string.digits, k=size))
        
        # Create a QR code object with a larger size and higher error correction
        qr = qrcode.QRCode(version=3, box_size=20, border=10, error_correction=qrcode.constants.ERROR_CORRECT_H)

        # Define the data to be encoded in the QR code
        data = rand_text

        # Add the data to the QR code object
        qr.add_data(data)

        # Make the QR code
        qr.make(fit=True)

        # Create an image from the QR code with a black fill color and white background
        img = qr.make_image(fill_color="black", back_color="white")

        # Save the QR code image
        img.save("qr_code.png")
        return rand_text,True
    else:
        return False,False

def scanner_qr():
    
    window_name = 'OpenCV QR Code'

    qcd = cv2.QRCodeDetector()
    
    scan = cv2.VideoCapture(0)

    try:
        while(True):
            ret2, frame2 = scan.read()
            if ret2:
                ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame2)
                if ret_qr:
                    for s, p in zip(decoded_info, points):
                        if s:
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                        frame2 = cv2.polylines(frame2, [p.astype(int)], True, color, 8)
                cv2.imshow(window_name, frame2)

            if cv2.waitKey(20) & 0xFF == ord('d'):
                break
        scan.release()
        cv2.destroyAllWindows() 
        return s

    except:
        return False
    
    
#secirity side
print("Security side:\n")
print("vehicle checking...")

if(text == Database and identity == True):
    print("Database and identity matched")
    print("access granted")
elif(text == Database and identity == False):
    print("Database matched but identity not matched")
    print("checking for QR")
    fir_text,flag = qrcheck()
    if(flag == True):
        print("QR present")
        print("Scanning qr code...")
        sec_val = scanner_qr()
        # print(sec_val)
        # print(fir_text)
        if(sec_val==fir_text):
            print("QR matched")
            print("Access granted")
        else:
            print("QR not matched")
            print("not-allowed")    
    else:
        print("QR not present")
        print("not-allowed")
        
elif(text != Database and identity == False):
    print("Database and identity both not matched")
    print("Register by security")
    print("access-granted")
    
else:
    print("not-allowed")