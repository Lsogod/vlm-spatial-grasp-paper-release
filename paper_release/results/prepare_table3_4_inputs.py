import json
from pathlib import Path
import numpy as np
import cv2

src=Path(r'c:\Users\16551\Desktop\html_demo\paper_release_bundle\data\dist_all')
out=Path(r'c:\Users\16551\Desktop\html_demo\paper_release_bundle\results\table3_4_inputs')
img_dir=out/'images'
mg_dir=out/'masks_grasp'
mp_dir=out/'masks_place'
for d in [img_dir,mg_dir,mp_dir]:
    d.mkdir(parents=True,exist_ok=True)

count=0
for npz_path in sorted(src.glob('*.npz')):
    stem=npz_path.stem
    data=np.load(npz_path,allow_pickle=True)
    if 'image' not in data.files:
        continue
    img=data['image']
    if img.ndim!=3:
        continue
    H,W=img.shape[:2]

    traj=None
    if 'trajectories' in data.files:
        raw=data['trajectories']
        if isinstance(raw,np.ndarray):
            if raw.ndim==0:
                raw=raw.item()
            elif raw.ndim==1 and raw.size==1:
                raw=raw[0]
            else:
                raw=raw.reshape(-1)[0]
        try:
            tlist=json.loads(str(raw))
            if isinstance(tlist,list) and tlist:
                traj=tlist[0]
        except Exception:
            traj=None

    mg=np.zeros((H,W),dtype=np.uint8)
    mp=np.zeros((H,W),dtype=np.uint8)

    if isinstance(traj,dict):
        def paint(regions,target):
            if not isinstance(regions,list):
                return
            for r in regions:
                coords=r.get('painted_coords',[])
                if not coords:
                    continue
                arr=np.array(coords,dtype=np.int32)
                if arr.ndim==2 and arr.shape[1]==2:
                    ys=np.clip(arr[:,0],0,H-1)
                    xs=np.clip(arr[:,1],0,W-1)
                    target[ys,xs]=255
        paint(traj.get('grasp_regions',[]),mg)
        paint(traj.get('brush_regions',[]),mg)
        paint(traj.get('place_regions',[]),mp)

    cv2.imwrite(str(img_dir/f'{stem}.png'), img)
    cv2.imwrite(str(mg_dir/f'{stem}.png'), mg)
    cv2.imwrite(str(mp_dir/f'{stem}.png'), mp)
    count+=1

print('prepared',count,'samples')
