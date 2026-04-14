# PythonSampleMouse.py
# Simulates OptiTrack marker data using mouse coordinates.
# Sends 3 markers (shoulder, elbow, hand) over UDP to Unity.
# The hand marker follows the mouse; elbow is interpolated; shoulder is fixed.
#
# No external dependencies — uses only Python standard library (tkinter + socket).

import socket
import time
import threading
import tkinter as tk

# --- Configuration ---
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005
SEND_RATE_HZ = 100  # Match OptiTrack's 100 FPS

# Marker IDs (mimicking OptiTrack labeled marker IDs)
MARKER_ID_SHOULDER = 7251
MARKER_ID_ELBOW    = 7252
MARKER_ID_HAND     = 7253

# Fixed shoulder position in world coordinates (meters)
SHOULDER_POS = [0.0, 1.4, 0.6]

# Window size for mouse capture
WINDOW_W = 800
WINDOW_H = 600

# World-space mapping range for the hand marker
# Mouse X (0..WINDOW_W) -> World X (-0.5 .. 0.5)
# Mouse Y (0..WINDOW_H) -> World Y (0.8 .. 1.4)  (table height range)
HAND_X_MIN, HAND_X_MAX = -0.5, 0.5
HAND_Y_MIN, HAND_Y_MAX =  0.8, 1.4
HAND_Z = 0.6  # Fixed Z (on the table plane)

# Visual constants
BG_COLOR      = "#1e1e1e"
TEXT_COLOR    = "#dcdcdc"
SHOULDER_COLOR = "#ff5050"
ELBOW_COLOR    = "#ffb432"
HAND_COLOR     = "#50ff50"
BONE_COLOR     = "#c8c8c8"
MARKER_RADIUS  = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_range(value, in_min, in_max, out_min, out_max):
    """Linearly maps value from [in_min, in_max] to [out_min, out_max]."""
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def lerp(a, b, t):
    """Linear interpolation between two 3D points."""
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def build_udp_message(frame_id, markers):
    """
    Build a UDP message string.
    Format: frame_id;marker_count;id,x,y,z;id,x,y,z;...
    Coordinates are formatted to 6 decimal places.
    """
    parts = [str(frame_id), str(len(markers))]
    for marker_id, pos in markers:
        parts.append(f"{marker_id},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}")
    return ";".join(parts)


def world_to_screen(wx, wy):
    """Convert world XY to canvas pixel coordinates."""
    cx = int(map_range(wx, HAND_X_MIN, HAND_X_MAX, 0, WINDOW_W))
    cy = int(map_range(wy, HAND_Y_MAX, HAND_Y_MIN, 0, WINDOW_H))  # invert Y
    return cx, cy


def draw_circle(canvas, x, y, r, color):
    canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class SimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OptiTrack Mouse Simulator — Move mouse to control hand marker")
        self.root.resizable(False, False)

        # State shared between UI and UDP threads
        self.mouse_x = WINDOW_W // 2
        self.mouse_y = WINDOW_H // 2
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP target: {UNITY_IP}:{UNITY_PORT}")
        print(f"Sending at {SEND_RATE_HZ} Hz")
        print(f"Markers: Shoulder={MARKER_ID_SHOULDER}, Elbow={MARKER_ID_ELBOW}, Hand={MARKER_ID_HAND}")
        print("Move the mouse inside the window to simulate hand movement.")
        print("Press Q or close the window to quit.\n")

        # Canvas for visualisation
        self.canvas = tk.Canvas(
            root, width=WINDOW_W, height=WINDOW_H,
            bg=BG_COLOR, highlightthickness=0
        )
        self.canvas.pack()

        # Info label below the canvas
        self.info_var = tk.StringVar(value="")
        info_label = tk.Label(
            root, textvariable=self.info_var,
            bg="#111111", fg=TEXT_COLOR,
            font=("Consolas", 11), justify="left", anchor="w",
            padx=10, pady=6
        )
        info_label.pack(fill="x")

        # Bind events
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.root.bind("<q>", lambda _: self.stop())
        self.root.bind("<Q>", lambda _: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        # Pre-create canvas objects so we only update them (faster than redraw)
        self._bone1  = self.canvas.create_line(0, 0, 0, 0, fill=BONE_COLOR, width=2)
        self._bone2  = self.canvas.create_line(0, 0, 0, 0, fill=BONE_COLOR, width=2)
        r = MARKER_RADIUS
        self._oval_s = self.canvas.create_oval(0,0,r,r, fill=SHOULDER_COLOR, outline=SHOULDER_COLOR)
        self._oval_e = self.canvas.create_oval(0,0,r,r, fill=ELBOW_COLOR,    outline=ELBOW_COLOR)
        self._oval_h = self.canvas.create_oval(0,0,r,r, fill=HAND_COLOR,     outline=HAND_COLOR)

        # Start UDP sender thread
        self._udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._udp_thread.start()

        # Start UI update loop
        self._update_ui()

    # ------------------------------------------------------------------
    def _on_mouse_move(self, event):
        with self.lock:
            self.mouse_x = event.x
            self.mouse_y = event.y

    # ------------------------------------------------------------------
    def _compute_positions(self):
        with self.lock:
            mx, my = self.mouse_x, self.mouse_y

        hand_x = map_range(mx, 0, WINDOW_W, HAND_X_MIN, HAND_X_MAX)
        hand_y = map_range(my, WINDOW_H, 0, HAND_Y_MIN, HAND_Y_MAX)  # invert Y
        hand_pos  = [hand_x, hand_y, HAND_Z]
        elbow_pos = lerp(SHOULDER_POS, hand_pos, 0.5)
        return mx, my, hand_pos, elbow_pos

    # ------------------------------------------------------------------
    def _udp_loop(self):
        """Runs in a background thread; sends UDP at SEND_RATE_HZ."""
        interval = 1.0 / SEND_RATE_HZ
        while self.running:
            t0 = time.perf_counter()
            _, _, hand_pos, elbow_pos = self._compute_positions()
            markers = [
                (MARKER_ID_SHOULDER, SHOULDER_POS),
                (MARKER_ID_ELBOW,    elbow_pos),
                (MARKER_ID_HAND,     hand_pos),
            ]
            msg = build_udp_message(self.frame_id, markers)
            self.sock.sendto(msg.encode("utf-8"), (UNITY_IP, UNITY_PORT))
            self.frame_id += 1

            elapsed = time.perf_counter() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    def _update_ui(self):
        """Scheduled on the tkinter main loop at ~60 fps for rendering."""
        if not self.running:
            return

        mx, my, hand_pos, elbow_pos = self._compute_positions()

        # Compute screen coords
        sx, sy = world_to_screen(SHOULDER_POS[0], SHOULDER_POS[1])
        ex, ey = world_to_screen(elbow_pos[0],    elbow_pos[1])
        hx, hy = world_to_screen(hand_pos[0],     hand_pos[1])
        r = MARKER_RADIUS

        # Update bones
        self.canvas.coords(self._bone1, sx, sy, ex, ey)
        self.canvas.coords(self._bone2, ex, ey, hx, hy)

        # Update marker ovals
        self.canvas.coords(self._oval_s, sx-r, sy-r, sx+r, sy+r)
        self.canvas.coords(self._oval_e, ex-r, ey-r, ex+r, ey+r)
        self.canvas.coords(self._oval_h, hx-r, hy-r, hx+r, hy+r)

        # Update info text
        sp = SHOULDER_POS
        ep = elbow_pos
        hp = hand_pos
        self.info_var.set(
            f"Frame: {self.frame_id}   Mouse: ({mx}, {my})   UDP → {UNITY_IP}:{UNITY_PORT}\n"
            f"Shoulder [{MARKER_ID_SHOULDER}]: ({sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f})   "
            f"Elbow [{MARKER_ID_ELBOW}]: ({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})   "
            f"Hand [{MARKER_ID_HAND}]: ({hp[0]:.3f}, {hp[1]:.3f}, {hp[2]:.3f})   "
            f"  |  Press Q to quit"
        )

        # Schedule next frame (~60 fps is plenty for the UI)
        self.root.after(16, self._update_ui)

    # ------------------------------------------------------------------
    def stop(self):
        self.running = False
        self.sock.close()
        self.root.destroy()
        print("Simulator stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()