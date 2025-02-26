from main import person_detection

# Define video sources & sound files
video_sources = [0, 1]

# Run each camera detection in parallel
if __name__ == "__main__":
    import multiprocessing

    processes = []
    for source in video_sources:
        p = multiprocessing.Process(target=person_detection, args=(source,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
