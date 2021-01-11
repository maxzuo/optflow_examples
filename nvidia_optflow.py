import argparse
import time

import numpy as np
import cv2

from sklearn.cluster import KMeans

from optflow import viz, flow as flw, util

def nvidia_optflow(videopath, clusters=4, frame_distance=1, perf_preset=10, display=False):
    cap = cv2.VideoCapture(videopath)

    n_flow = flw.FarnebackFlowIterator(cap, frame_distance=frame_distance, perfPreset=10)

    flows = []

    op_start_time = time.time()
    print("\n========\nCalculating optical flow\n========")
    
    for flow, frame in util.tqdm(n_flow):
        mag, ang = flw.to_polar(flow)

        if display:
            cv2.imshow('frame', frame)
            cv2.imshow('flow', viz.flowHSV(flow))
        
        flows.append([np.sum(mag), np.average(ang, weights=mag)])

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

    kmeans_start_time = time.time()
    print(f"{kmeans_start_time - op_start_time:.3f}s")
    print("\n========\nPerforming kMeans clustering\n========")

    flows = np.asarray(flows)
    min_flow = np.min(flows, axis=0)
    max_flow = np.max(flows, axis=0)

    flows = (flows - min_flow) / (max_flow - min_flow)

    kM = KMeans(n_clusters=clusters, random_state=101, n_init=100).fit(flows)

    print(f"{time.time() - kmeans_start_time:.3f}s")

    return flows, (kM.cluster_centers_, kM.labels_)

def write_video(videopath:str, outfile:str, flows, centers, labels, k=4, frame_distance=1, display=False):
    cap = cv2.VideoCapture(videopath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(3)
    height = cap.get(4)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(outfile, fourcc, fps, (int(width), int(height)))

    for _ in range(frame_distance): cap.read()

    print("\n========\nPerforming kMeans clustering\n========")
    for plot in viz.display_kmeans(flows, labels, centers):
        _, frame = cap.read()
        plot = cv2.resize(plot, None, fx=0.5, fy=0.5)

        frame[0:plot.shape[0], 0:plot.shape[1]] = plot
        out.write(frame)
        if display:
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    
    out.release()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--video", help="path to video file", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="path to output file", type=str, default=None)
    parser.add_argument("-f", "--frame_distance", help="frame distance for optical flow calculation", type=int, default=1)
    parser.add_argument("-k", "--kclusters", help="number of clusters to use", type=int, default=4)
    parser.add_argument('-d', "--display", help="display optical flow and states", action="store_true", default=False)

    args = parser.parse_args()


    flows, (centers, cluster_assignments) = nvidia_optflow(args.video, args.kclusters, perf_preset=20, frame_distance=args.frame_distance, display=args.display)


    np.save("centers.npy", centers)
    np.save("labels.npy", cluster_assignments.astype(np.uint8))

    if args.outfile:
        write_video(args.video, args.outfile, flows, centers, cluster_assignments, k=4, display=args.display)

    cv2.destroyAllWindows()