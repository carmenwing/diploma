from __future__ import print_function
import sys
import json
import os.path
import math
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255)]
#algoritmi de urmărire
TRACKER_TYPES = {
  'BOOSTING': cv2.TrackerBoosting,
  'MIL': cv2.TrackerMIL,
  'KCF': cv2.TrackerKCF,
  'TLD': cv2.TrackerTLD,
  'MEDIANFLOW': cv2.TrackerMedianFlow,
  'GOTURN': cv2.TrackerGOTURN,
  'MOSSE':  cv2.TrackerMOSSE,
}

class LoggedObject:
    DEBUG = False #
    def debug(self, *args):
        if self.DEBUG:
            print(*args)

class Settings: #setări preluare date din fișier de configurare
    def __init__(self, config_file=None):
        try:
            with open(config_file) as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            print("Error loading config file:", str(e)) #eroare la încărcare
            sys.exit()
        except Exception as e:
            self.config = {}

    @property
    def ALWAYS_FACE_DETECTION(self):
        #detecția se face sau nu
        return not bool(self.config.get("do_tracking", False))

    @property
    def MAX_TRACK_SECONDS(self):
        #câte secunde se face urmărire, fără detecție
        return self.config.get("max_track_seconds", 0.3)

    @property
    def KEEP_PERSON_SECONDS(self):
        #după câte secunde o față devine istorică
        return self.config.get("keep_person_seconds", 5)

    @property
    def CASCADE_FILENAME(self):
        #calea către fișierul cascadă utilizat
        program_dir = os.path.dirname(os.path.abspath(__file__))
        default_cascade = os.path.join(program_dir, 'haarcascade_frontalface_default.xml')
        return self.config.get("cascade_filename", default_cascade)

    @property
    def SCALE_FACTOR(self):
        #parametru detector - factor de scalare
        return self.config.get("scale_factor", 1.3)

    @property
    def MIN_NEIGHBORS(self):
        #parametru detector - nr minim vecini
        return self.config.get("min_neighbors", 4)

    @property
    def MIN_SIZE_X(self):
        #parametru detector - x minim
        return self.config.get("min_size_x", 50)

    @property
    def MIN_SIZE_Y(self):
        #parametru detector - y minim
        return self.config.get("min_size_y", 50)

    @property
    def TRACKER_TYPE(self):
        #algoritm urmărire prestabilit
        tracker_type = self.config.get("tracker_type", "MIL")
        if tracker_type not in TRACKER_TYPES:
            print("[ERROR]: Invalid tracker_type: %s" % tracker_type)
            tracker_type = "MIL"
        return tracker_type

    @property
    def DETECT_HEARTRATE(self):
        #calculează puls sau nu
        return self.config.get("detect_heartrate", False)

    @property
    def FONT(self):
        return cv2.FONT_HERSHEY_SIMPLEX



class Person(LoggedObject):
    COLOR_INDEX = 0

    MIN_HR = 30
    MAX_HR = 180

    def __init__(self, cntx, settings):
        self.display_fft = True
        self.cntx = cntx
        self.N = 250
        self.t0 = time.time() #moment de start
        self.means = []
        self.times = []
        self.magnitude = np.array([])
        self.freqs = np.array([])
        self.color = COLORS[Person.COLOR_INDEX]
        Person.COLOR_INDEX = (Person.COLOR_INDEX + 1) % len(COLORS)
        self.tracker_type = settings.TRACKER_TYPE
        self.font = settings.FONT
        self.last_fd_frame_time = cntx.last_fd_frame_time

    def roi(self, rectangle):
        x, y, w, h = rectangle
        return int(x+0.3*w), int(y+0.1*h), int(0.4*w), int(0.1*h)

    def roi_mean(self):
        x, y, w, h = self.roi(self.rectangle)
        rect_roi = self.cntx.g[y:y+h, x:x+w]
        return rect_roi.mean()

    def update_face(self, rectangle=None): #pe fiecare frame de fd recreez tracker si il initializez 
        if rectangle is not None:
            self.last_detection_time = time.time()
            self.rectangle = rectangle #save rect pt a sti ca este al pers resp
            self.tracker = TRACKER_TYPES[self.tracker_type].create()
            self.tracker.init(self.cntx.g, tuple(self.rectangle))
        self.add_timestamp()

    def track_face(self): 
        ok, rectangle = self.tracker.update(self.cntx.g)
        if ok:#daca tracking se face cu succes
            self.last_detection_time = time.time() #salveaza momentul de timp
            self.rectangle = tuple(map(int, rectangle)) 
        self.add_timestamp()

    def add_timestamp(self):
        self.times.append(time.time() - self.t0)
        self.times = self.times[-self.N:]
        self.debug("times: %d" % len(self.times))

    def calculate_means(self):
        self.means.append(self.roi_mean()) #formez un sir din ultima parte din vector si adaug un element
        self.means = self.means[-self.N:]
        self.debug("means: %d" % len(self.means))

    def calculate_hr(self):
        self.calculate_means()
        if len(self.means) < 10:
            return
        y = np.array(self.means, dtype=float)
        n = len(y) # length of the signal
        fps = float(n) / (self.times[-1] - self.times[0])
        even_times = np.linspace(self.times[0], self.times[-1], n)
        y = np.interp(even_times, self.times, y)
        y = np.hamming(n) * y #corelatie
        y = y - np.mean(y)
        raw = np.fft.rfft(y*2)
        fft = np.abs(raw)
        freqs = float(fps) / n * np.arange(n / 2 + 1)
        freqs = 60. * freqs
        idx = np.where((freqs > Person.MIN_HR) & (freqs < Person.MAX_HR))
        self.freqs = freqs[idx]
        self.magnitude = fft[idx]

    @property
    def heart_rate(self):
        if len(self.magnitude) < 10:
            return None
        max_idx = np.argmax(self.magnitude)
        return self.freqs[max_idx]

    def draw_widgets(self):
        x, y, w, h = self.rectangle #dreptunghiul de față
        cv2.rectangle(self.cntx.frame, (x, y), (x + w, y + h), self.color, 2)
        x1, y1, w1, h1 = self.roi(self.rectangle)# dreptunghiul regiunii de interes
        cv2.rectangle(self.cntx.frame, (x1, y1), (x1 + w1, y1 + h1), self.color, 2)
        if self.heart_rate:
            cv2.putText(self.cntx.frame, "HR:" + str(int(self.heart_rate)), (x + w - 80, y - 4), self.font, 0.8, self.color, 1, cv2.LINE_AA)
        if self.display_fft:
            freqs = (self.freqs - min(self.freqs)) * 200. / (max(self.freqs)- min(self.freqs))
            mag = self.cntx.video_height - self.magnitude * 100 / max(self.magnitude)
            pts = np.vstack((freqs, mag)).astype(np.int32).T
            cv2.polylines(self.cntx.frame, [pts], isClosed=False, color=self.color)

    def is_my_face(self, rect):
        l, t, w, h = self.rectangle
        r, b = l + w, t + h
        l1, t1, w1, h1 = rect
        r1, b1 = l1 + w1, t1 + h1
        is_mine = l < r1 and l1 < r and t < b1 and t1 < b
        if not is_mine:
            print(rect, "is not mine", self.rectangle)
        return is_mine

    @property
    def display_fft(self):
        return self._display_fft and len(self.freqs) > 10

    @display_fft.setter
    def display_fft(self, value):
        self._display_fft = bool(value)


class Program(LoggedObject):   #constructorul programului
    def __init__(self, settings): #în instanța programului salvez setările
        self.settings = settings
        self.font = settings.FONT
        self.face_detector = FaceDetector(self, settings) #creez detector de fețe
        self.skin_detector = SkinDetector(self)
        self.cap = cv2.VideoCapture(0) #se începe captura
        self.current_frame_time = None
        self.last_fd_frame_time = None
        self.last_frame_time = None
        self.persons = []
        self.timings_trk=[]
        self.timings_fd=[]

    def remove_persons(self):
        to_keep = []
        for person in self.persons:
            if self.current_frame_time - person.last_detection_time < self.settings.KEEP_PERSON_SECONDS:
                to_keep.append(person)
            else:
                print("Deleting", id(person))
        self.persons = to_keep

    def is_facedetection_frame(self): #se face detectie daca:
        if self.settings.ALWAYS_FACE_DETECTION:
            return True #este specificat în setări că se face mereu
        if len(self.persons) == 0 or self.last_fd_frame_time is None:
            return True #sau nu e nicio persoană deja detectată sau dacă sunt la primul cadru preluat
            #sau dacă a trecut un anumit timp de la ultima detecție
        return self.current_frame_time - self.last_fd_frame_time > self.settings.MAX_TRACK_SECONDS

    @property
    def video_height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def video_width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)


    def run(self):
        while(True):
            start = self.current_frame_time = time.time() #moment de timp la care se salvează cadrul
            ret, frame = self.cap.read()  #preluare cadru, caz în care ret = true
            frame = cv2.flip(frame, 1)
            if not ret: #dacă nu s-a preluat cadru de la camera web
                continue #rulează din nou bucla while

            self.frame = frame
            self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY))
            self.b, self.g, self.r = cv2.split(self.frame) #canalele de culoare
            self.skin_map = self.skin_detector.get_skin()
            #verific daca frame curent este de detecție sau tracking
            face_detection = self.is_facedetection_frame()
            self.debug(len(self.persons), "FaceDetection" if face_detection else "FaceTracking")

            if face_detection:
                faces = self.face_detector.get_faces() #iau toate fețele detectate
                self.last_fd_frame_time = self.current_frame_time
                is_already_a_person = [False] * len(faces) #vector initializat cu False
                for person in self.persons:#pentru fiecare persoană detectată
                    found_my_face = False 
                    for i, face in enumerate(faces): #verific ce față se potrivește
                        if not is_already_a_person[i] and person.is_my_face(face):#pentru fiecare persoană iau toate fețele
                            is_already_a_person[i] = True #notez că fața aparține deja unei persoane
                            person.update_face(face) #update cu noul dreptunghi de față
                            found_my_face = True #și notez că pentru persoana respectivă am găsit o față
                            break #dacă am găsit, mă opresc din căutat
                    if not found_my_face: #dacă nu am găsit fața (detectorul mai are erori)
                        self.debug("Updating my face with the previous frame", id(person)) #iau ultima față pe care o știam
                        person.update_face()#adaugă încă un element în vectorul de medii șî în cel de timp
                    #e nevoie să nu mă fi mișcat mult
                    #dacă am un id de persoană pentru care nu am găsit fața, refolosesc ultima față salvată
                    #în cazul în care nu a găsit persoana pentru că aceasta a ieșit din cadru, ea va fi eliminată la tracking
                    #pentru că va deveni persoană istorică
                for face, is_person in zip(faces, is_already_a_person):
                    if not is_person:#pentru fețele care nu sunt persoane
                        person = Person(self, self.settings)#creez o persoană
                        print("Creating a new person", id(person))
                        person.update_face(face)#update 
                        self.persons.append(person)#se adaugă persoana la lista de persoane
            else:
                # frame-ul current este un frame pe care se face object tracking
                for person in self.persons:
                    person.track_face()
            self.remove_persons()
            # Afiseaza frame-ul rezultat
            for person in self.persons: #pentru fiecare persoană
                if self.settings.DETECT_HEARTRATE:
                    person.calculate_hr()
                person.draw_widgets()#desenez dreptunghiul de față
            end = time.time() #măsoară cât a durat procesarea cadrului
            if face_detection:
                self.timings_fd.append((end-start)*1000)#adaug ori în timings de detecție
            else:
                self.timings_trk.append((end-start)*1000)#ori în timings de urmărire
            #cv2.putText(frame, str(int(1000*(end - self.current_frame_time))) + "ms/frame",(10,50), self.font, 0.8,(0,0,255), 1, cv2.LINE_AA)
            # cv2.putText(frame, "PERSOANE DETECTATE: " + str(len(self.persons)),(10,80), self.font, 0.8, (0,0,255), 1, cv2.LINE_AA)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        print("Mean time/frame with FD:", sum(self.timings_fd)/len(self.timings_fd))
        print("Mean time/frame with Tracking:", sum(self.timings_trk)/max(1, len(self.timings_trk)))


class SkinDetector(LoggedObject):
    def __init__(self, cntx):
        self.cntx = cntx

    def get_skin(self):
        # TODO: implement skin detection
        return self.cntx.frame

class FaceDetector(LoggedObject):
    def __init__(self, cntx, settings):
        self.cntx = cntx
        self.detector = cv2.CascadeClassifier(settings.CASCADE_FILENAME)
        self.scaleFactor = settings.SCALE_FACTOR
        self.minNeighbors = settings.MIN_NEIGHBORS
        self.minSize = (settings.MIN_SIZE_X,settings.MIN_SIZE_Y)

    def get_faces(self):#pentru implementare detecție de piele
        faces = []
        for face in self.detector.detectMultiScale(self.cntx.frame, scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize,
            flags=cv2.CASCADE_SCALE_IMAGE):
            if self.check_skin(face, self.cntx.skin_map):
                faces.append(face)
        return faces

    def check_skin(self, face, skin_map):#verific dacă zona conține suficientă piele
        # TODO: check if skin in face rectangle is > 60%
        return True


if __name__ == "__main__":
    app = Program(Settings("hr_config.json")) 
    app.run()