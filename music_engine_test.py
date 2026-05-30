import pygame.midi
import time
import math
import pygame

pygame.midi.quit()
pygame.midi.init()
player = pygame.midi.Output(pygame.midi.get_default_output_id())
player.set_instrument(0)  # 40 = violin, 0=piano, 73=flute, 19=organ

# --- Scales ---
SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "chromatic":  list(range(12)),
}
current_scale = "chromatic"
ROOT_NOTE = 48  # C3

def y_to_note(y_norm):
    """y_norm: 0.0 (top) → 1.0 (bottom). Top = high pitch."""
    scale = SCALES[current_scale]
    idx = int((1.0 - y_norm) * (len(scale) * 2))
    idx = max(0, min(idx, len(scale) * 2 - 1))
    octave, step = divmod(idx, len(scale))
    return ROOT_NOTE + octave * 12 + scale[step]

def get_finger_spread(hand_landmarks, w):
    index_x = hand_landmarks[8].x * w
    pinky_x = hand_landmarks[20].x * w
    return abs(index_x - pinky_x)

def is_fist(hand_landmarks):
    wrist_y = hand_landmarks[0].y
    tips = [hand_landmarks[i].y for i in [8, 12, 16, 20]]
    return all(tip > wrist_y + 0.05 for tip in tips)

def get_hand_tilt(hand_landmarks):
    """
    Returns tilt relative to upright (90°).
    Upright hand = 0, tilt left = negative, tilt right = positive.
    """
    wrist   = hand_landmarks[0]
    mid_mcp = hand_landmarks[9]
    dx = mid_mcp.x - wrist.x
    dy = mid_mcp.y - wrist.y
    angle = math.degrees(math.atan2(-dy, dx))
    return angle - 90   # subtract 90 so upright = 0

def play_chord(note, volume):
    notes = [note, note + 4, note + 7]   # root + 3rd + 5th
    for n in notes:
        player.note_on(n, volume)
    return notes
         
class MusicConductor:
    def __init__(self):
        self.last_note = -1
        self.last_note_time = 0
        self._last_x = 0.5
        self.velocity_history = []
        self.note_interval = 0.4
        self.volume = 80
        self.active_notes = []
        self.hand_visible = False
        self._fist_held = False

    def send_pitch_bend(self, bend_value):
        """
        bend_value: -8192 to +8191, 0 = neutral.
        """
        midi_bend = bend_value + 8192          # shift to 0–16383
        midi_bend = max(0, min(16383, midi_bend))
        lsb = midi_bend & 0x7F
        msb = (midi_bend >> 7) & 0x7F
        channel = 0
        player.write([[[0xE0 | channel, lsb, msb], 0]])

    def apply_vibrato(self, tilt_angle):
        """
        Maps hand tilt → pitch bend.
        Dead zone of ±15° so small wobbles don't bend pitch.
        """
        dead_zone = 15.0
        max_tilt  = 60.0    # angle at which full bend is reached
        max_bend  = 4000    # semitone-and-a-bit range (8192 = full ±2 semitones)

        if abs(tilt_angle) < dead_zone:
            bend = 0
        else:
            # sign-preserving scale outside dead zone
            sign = 1 if tilt_angle > 0 else -1
            scaled = (abs(tilt_angle) - dead_zone) / (max_tilt - dead_zone)
            scaled = min(1.0, scaled)
            bend = int(sign * scaled * max_bend)

        self.send_pitch_bend(bend)
        return bend

    def update_volume_hand(self, kx, ky, frame_w, frame_h, hand_landmarks):
        """Left hand: controls volume + scale cycling."""
        global current_scale
        y_norm = ky / frame_h
        self.volume = int((1.0 - y_norm) * 100) + 20
        self.volume = max(20, min(127, self.volume))

        if is_fist(hand_landmarks):
            if not self._fist_held:
                idx = SCALE_ORDER.index(current_scale)
                current_scale = SCALE_ORDER[(idx + 1) % len(SCALE_ORDER)]
                self._fist_held = True
                print(f"Scale switched to: {current_scale}")
        else:
            self._fist_held = False

    def update_pitch_hand(self, kx, ky, frame_w, frame_h, hand_landmarks):
        """Right hand: controls pitch + tempo + chord/note triggering."""
        y_norm = ky / frame_h
        x_norm = kx / frame_w

        dx = abs(x_norm - self._last_x)
        self._last_x = x_norm
        self.velocity_history.append(dx)
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)
        avg_speed = sum(self.velocity_history) / len(self.velocity_history)
        self.note_interval = 0.6 - avg_speed * 8
        self.note_interval = max(0.15, min(0.6, self.note_interval))

        now = time.time()
        if now - self.last_note_time > self.note_interval:
            note = y_to_note(y_norm)
            self.stop_all()
            spread = get_finger_spread(hand_landmarks, frame_w)
            if spread > 80:
                self.active_notes = play_chord(note, self.volume)
            else:
                player.note_on(note, self.volume)
                self.active_notes = [note]
            self.last_note = note
            self.last_note_time = now
        tilt = get_hand_tilt(hand_landmarks)
        bend = self.apply_vibrato(tilt)
        return bend

    def stop_all(self):
        for note in self.active_notes:
            player.note_off(note, 0)
        if self.last_note != -1:
            player.note_off(self.last_note, 0)
        self.active_notes = []
        self.last_note = -1

conductor = MusicConductor()
print("Music engine ready!")
