CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def print_progress(progress, first, width=50):

    if not first:
        print(CURSOR_UP_ONE + ERASE_LINE, end="")

    done = int(progress * width)
    not_done = width - done
    percent = int(100 * progress)

    print("[{}{}] {:3}%".format("#" * done, "-" * not_done, percent))
