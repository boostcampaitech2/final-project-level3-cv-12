from PIL import Image
from PIL import ImageDraw
import numpy as np
from oauth2client.client  import GoogleCredentials
import sys
import io
import os
import json
import base64
from genericpath import isfile
import hashlib
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery

NUM_THREADS = 10
MAX_FACE = 2
MAX_LABEL = 50
IMAGE_SIZE = 128,128
MAX_ROLL = 30
MAX_TILT = 30
MAX_PAN = 30

def pil_draw_rect(image, lt, rb):
    point1 = (lt['x'], lt['y'] )
    point2 = (rb['x'], rb['y'] )
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=(255, 0, 255), width=3)

    return image

def pil_draw_cropped_rect(image, boundingPoly,lt): #face, eye
    x_len = boundingPoly[1]['x']- boundingPoly[0]['x']
    y_len = boundingPoly[2]['y']- boundingPoly[1]['y']
    x_div = x_len / 128
    y_div = y_len / 128
    point1 = ((lt['x']- boundingPoly[0]['x'])/x_div), ((lt['y'] - boundingPoly[0]['y'])/y_div)
    point2 = (point1[0]+1, point1[1]+1 )
    print(point1,point2)
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=(255, 0, 255), width=5)

    return image

def point_for_128( boundingPoly,lt):
    x_len = boundingPoly[1]['x']- boundingPoly[0]['x']
    y_len = boundingPoly[2]['y']- boundingPoly[1]['y']
    x_div = x_len / 128
    y_div = y_len / 128

    point = ((lt['x']- boundingPoly[0]['x'])/x_div), ((lt['y'] - boundingPoly[0]['y'])/y_div)

    pointx4 = (((lt['x']- boundingPoly[0]['x'])/x_div)*4), (((lt['y'] - boundingPoly[0]['y'])/y_div)*4)

    return point, pointx4

def point_for_128_face( boundingPoly,fdBoundingPoly):
    point = []
    pointx4 = []
    for i in range(4):
        x_len = boundingPoly[1]['x']- boundingPoly[0]['x']
        y_len = boundingPoly[2]['y']- boundingPoly[1]['y']
        x_div = x_len / 128
        y_div = y_len / 128

        point.append((((fdBoundingPoly[i]['x']- boundingPoly[0]['x'])/x_div), ((fdBoundingPoly[i]['y'] - boundingPoly[0]['y'])/y_div)))

        pointx4.append(((((fdBoundingPoly[i]['x']- boundingPoly[0]['x'])/x_div)*4), (((fdBoundingPoly[i]['y'] - boundingPoly[0]['y'])/y_div)*4)))

    return point, pointx4
    

 

# index to transfrom image string label to number
global_label_index = 0 
global_label_number = [0 for x in range(1000)]
global_image_hash = []
global face_ann 
 
def json_maker(json1, json2, f, face_ann):
    left_eye,left_eyex4 = point_for_128(face_ann['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices'], face_ann['responses'][0]['faceAnnotations'][0]['landmarks'][0]['position'])
    right_eye,right_eyex4 = point_for_128(face_ann['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices'], face_ann['responses'][0]['faceAnnotations'][0]['landmarks'][1]['position'])
    nose,nosex4 = point_for_128(face_ann['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices'], face_ann['responses'][0]['faceAnnotations'][0]['landmarks'][7]['position'])
    mouth, mouthx4 = point_for_128(face_ann['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices'], face_ann['responses'][0]['faceAnnotations'][0]['landmarks'][12]['position'])
    face_fit,face_fitx4 = point_for_128_face(face_ann['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices'], face_ann['responses'][0]['faceAnnotations'][0]['fdBoundingPoly']['vertices'])

    face = dict()
    face['left_eye'] = left_eye
    face['right_eye'] = right_eye
    face['nose'] = nose
    face['mouth'] = mouth
    face['face_fit'] = face_fit
    json1[f] = face
    
    face2 = dict()
    face2['left_eye'] = left_eyex4
    face2['right_eye'] = right_eyex4
    face2['nose'] = nosex4
    face2['mouth'] = mouthx4
    face2['face_fit'] = face_fitx4
    json2[f] = face2

    return json

 

 

class FaceDetector():
    def __init__(self):
        # initialize library
        #credentials = GoogleCredentials.get_application_default()
        scopes = ['https://www.googleapis.com/auth/cloud-platform']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
                './savvy-motif-332014-60fbd7f4593b.json', scopes=scopes)
        self.service = discovery.build('vision', 'v1', credentials=credentials)
        #print ("Getting vision API client : %s" ,self.service)

    #def extract_face(selfself,image_file,output_file):

    def skew_angle(self):
        return None
    
    def detect_face(self,image_file):
        global face_ann
        try:
            with io.open(image_file,'rb') as fd:
                image = fd.read()
                batch_request = [{
                        'image':{
                            'content':base64.b64encode(image).decode('utf-8')
                            },
                        'features':[
                            {
                            'type':'FACE_DETECTION',
                            'maxResults':MAX_FACE,
                            },
                            {
                            'type':'LABEL_DETECTION',
                            'maxResults':MAX_LABEL,
                            }
                                    ]
                        }]
            fd.close()

            request = self.service.images().annotate(body={'requests':batch_request, })
            response = request.execute()
            if 'faceAnnotations' not in response['responses'][0]:
                print('[Error] %s: Cannot find face ' % image_file)
                return None
                
            face = response['responses'][0]['faceAnnotations']
            face_ann = response
            label = response['responses'][0]['labelAnnotations']
          
            if len(face) > 1 :
                print('[Error] %s: It has more than 2 faces in a file' % image_file)
                return None
            
            roll_angle = face[0]['rollAngle']
            pan_angle = face[0]['panAngle']
            tilt_angle = face[0]['tiltAngle']
            angle = [roll_angle,pan_angle,tilt_angle]
            
            # check angle
            # if face skew angle is greater than > 20, it will skip the data
            if abs(roll_angle) > MAX_ROLL or abs(pan_angle) > MAX_PAN or abs(tilt_angle) > MAX_TILT:
                print('[Error] %s: face skew angle is big' % image_file)
                return None
            
            # check sunglasses
            for l in label:
                if 'sunglasses' in l['description']:
                  print('[Error] %s: sunglass is detected' % image_file)  
                  return None
            
            for l in label:
                if 'glasses' in l['description']:
                  print('[Error] %s: glass is detected' % image_file)  
                  return None
            
            
            box = face[0]['boundingPoly']['vertices']
            left = box[0]['x']
            try:
                top = box[1]['y']
            except:
                top = 0
                
            right = box[2]['x']
            bottom = box[2]['y']

                

            rect = [left,top,right,bottom]

                

            print("[Info] %s: Find face from in position %s and skew angle %s" % (image_file,rect,angle))

            return rect

        except Exception as e:

            print('[Error] %s: cannot process file : %s' %(image_file,str(e)) )

            

    def rect_face(self,image_file,rect,outputfile):

        try:

            fd = io.open(image_file,'rb')

            image = Image.open(fd)

            draw = ImageDraw.Draw(image)

            draw.rectangle(rect,fill=None,outline="green")

            image.save(outputfile)

            fd.close()

            print('[Info] %s: Mark face with Rect %s and write it to file : %s' %(image_file,rect,outputfile) )

        except Exception as e:

            print('[Error] %s: Rect image writing error : %s' %(image_file,str(e)) )

        

    def crop_face(self,image_file,rect,outputfile):

        

        global global_image_hash

        try:

            fd = io.open(image_file,'rb')

            image = Image.open(fd)  

 

            # extract hash from image to check duplicated image

            m = hashlib.md5()

            with io.BytesIO() as memf:

                image.save(memf, 'PNG')

                data = memf.getvalue()

                m.update(data)

            image_hash = m.hexdigest()

            

            if image_hash in global_image_hash:

                print('[Error] %s: Duplicated image' %(image_file) )

                return None

            global_image_hash.append(image_hash)

 

            crop = image.crop(rect)

            im = crop.resize(IMAGE_SIZE,Image.ANTIALIAS)

            

            

            im.save(outputfile,"JPEG")

            fd.close()

            print('[Info]  %s: Crop face %s and write it to file : %s' %( image_file,rect,outputfile) )

            return True

        except Exception as e:

            print('[Error] %s: Crop image writing error : %s' %(image_file,str(e)) )

        

    def getfiles(self,src_dir):

        files = []

        for f in os.listdir(src_dir):

 

            if os.path.splitext(f)[1] == ".jpg" or os.path.splitext(f)[1] == ".jpeg" or os.path.splitext(f)[1] == ".png":

                if isfile(os.path.join(src_dir,f)):

                    if not f.startswith('.'):

                         files.append(os.path.join(src_dir,f))

            else:

                pass

        return files

    

    # read files in src_dir and generate image that rectangle in face and write into files in des_dir

    def rect_faces_dir(self,src_dir,des_dir):

        if not os.path.exists(des_dir):

            os.makedirs(des_dir)

            

        files = self.getfiles(src_dir)

        for f in files:

            des_file = os.path.join(des_dir,os.path.basename(f))

            rect = self.detect_face(f)

            if rect != None:

                self.rect_face(f, rect, des_file)

    

    # read files in src_dir and crop face only and write it into des_dir

    def crop_faces_dir(self,src_dir,des_dir,maxnum):

        global face_ann

        print('2')

        des_dir_annotationsx4 = os.path.join(src_dir,'annotationsx4')

        des_dir_annotations = os.path.join(des_dir,'annotations')

        path,folder_name = os.path.split(src_dir)

        label = folder_name

        

        # create label file. it will contains file location 

        # and label for each file

        annotationsx4_file = open(src_dir+'/annotationsx4'+'/annotationsx4_file.txt','a')

        annotation_file = open(src_dir+'/annotations'+'/annotations_file.txt','a')

        print('3')

        global global_label_index

        cnt = 0 

        num = 0 # number of training data

        all_json =dict()

        all_jsonx4 =dict()
        src_dir = os.path.join(src_dir, 'images')
        files = os.listdir(src_dir)
                
        for f in files:
            f = os.path.join(src_dir,f)
            print('@@',f)
            rect = self.detect_face(f)

 

            # replace ',' in file name to '.'

            # because ',' is used for deliminator of image file name and its label

            des_file_name = os.path.basename(f)
            des_file_name = des_file_name.replace(',','_')

          

            if rect != None:

                # 70% of file will be stored in training data directory

                des_file = os.path.join(des_dir,des_file_name)

                    # if we already have duplicated image, crop_face will return None

                if self.crop_face(f, rect, des_file ) != None:

                    try:
                        json_temp = json_maker(all_json, all_jsonx4, os.path.basename(f), face_ann)

                        new_face_ann = json.dumps(all_json, indent=2)
                        new_face_annx4 = json.dumps(all_jsonx4, indent=2) # x4

                        num = num + 1

                        global_label_number[global_label_index] = num

                        print('training', num)

                    except:

                        pass

 


        annotation_file.write("%s" %(new_face_ann))

        annotationsx4_file.write("%s" %(new_face_annx4))

 

        global_label_index = global_label_index + 1 

        print('## label %s has %s of training data' %(global_label_index,num))

        annotationsx4_file.close()

        annotation_file.close()

        

    def getdirs(self,dir):

        dirs = []

        for f in os.listdir(dir):

            f=os.path.join(dir,f)

            if not f.startswith('.'):

                dirs.append(f)

 

        return dirs

        

    def crop_faces_rootdir(self,src_dir,des_dir,maxnum):

        # crop file from sub-directoris in src_dir

        self.crop_faces_dir(os.path.join(src_dir), des_dir,maxnum)

        #loop and run face crop

        global global_label_number

        print("number of datas per label ", global_label_number)

 

#usage

# arg[1] : src directory

# arg[2] : destination diectory

# arg[3] : max number of samples per class        

def main(argv):

    srcdir= '/home/myamya/project/image_files'

    desdir = '/home/myamya/project/image_files/cropped_image'

    maxnum = 10000

    

    detector = FaceDetector()

    print('0')

 

    detector.crop_faces_rootdir(srcdir, desdir, maxnum)

    #detector.crop_faces_dir(inputfile,outputfile)

    #rect = detector.detect_face(inputfile)

    #detector.rect_image(inputfile, rect, outputfile)

    #detector.crop_face(inputfile, rect, outputfile)

    

if __name__ == "__main__":

    main(sys.argv)


