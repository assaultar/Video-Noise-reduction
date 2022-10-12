from glob import glob
import os
import shutil
import ffmpeg
import cv2

def pre_image(args):

    if not args['in_image']:
        return args
    
    img = args['in_image']
    if not os.path.exists(img):
        print('Image Not Found')
        exit()

    pref = '.video_cache/tmp/'

    if os.path.exists(pref):
        shutil.rmtree(pref)

    os.makedirs(pref)
    shutil.copy(img, pref + "img_0.png")
    shutil.copy(img, pref + "img_1.png")
    shutil.copy(img, pref + "img_2.png")
    args['in_path'] = pref
    args['out_path'] = pref + '/out/'
    args['in_video'] = args['out_video'] = None
    return args

def post_image(args):
    
    if not args['out_image']:
        return
    
    pref = glob('.video_cache/tmp/out/*.png')
    if not args['out_image']:
        shutil.copy(pref[0], "./out.png")
    
    shutil.copy(pref[0], args['out_image'])
    shutil.rmtree('.video_cache/tmp')

def pre_process(args):
 
    if not args['in_video']:
        return args

    video = args['in_video']
    if not os.path.exists(video):
        print("Video Not Found")
        exit(1)

    if not os.path.exists(".video_cache"):
        os.makedirs(".video_cache")

    vname = os.path.basename(video)
    dir_template = f".video_cache/{vname}_IN/"
    if os.path.exists(dir_template):
        shutil.rmtree(dir_template)
    os.makedirs(dir_template)

    v0 = cv2.VideoCapture(video)
    now = 0
    writeT = dir_template + "I"
    while v0.isOpened():
        ret, Frame = v0.read()
        if not ret:
            break
        cv2.imwrite(writeT + now.__str__() + ".jpg", Frame)
        now += 1
    v0.release()
    cv2.destroyAllWindows()

    args['in_path'] = dir_template
    if args['out_video']:
        args['out_path'] = f".video_cache/{vname}_out/"
    return args

def post_process(args):

    if not args['out_video']:
        return args

    out_video = args["out_video"]
    if os.path.exists(out_video):
        shutil.move(out_video, f"{out_video}_old")

    op = args['in_path']                        \
        or args['in_video']                     \
        or print('Error: No output type given') \
        and exit(2)

    vname = os.path.basename(op)
    opath = f"{args['out_path']}/*.png" \
        or f".video_cache/{vname}_out/*.png"

    ffmpeg.input(
        opath
        , pattern_type='glob'
        , framerate=20).output(out_video).run()