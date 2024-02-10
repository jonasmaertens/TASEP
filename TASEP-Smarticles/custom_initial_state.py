from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from smarttasep import EnvParams, Hyperparams, Trainer


def text_phantom(text):
    # Availability is platform dependent
    font = '/Library/Fonts/Arial Unicode.ttf'

    # Create font
    pil_font = ImageFont.truetype(font, size=300 // len(text),
                                  encoding="unic")
    _, _, text_width, text_height = pil_font.getbbox(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [100, 100], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((215 - text_width) // 2,
              (50 - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return ((255 - np.asarray(canvas)) / 255.0)[:, :, 0]


initial_state = text_phantom("SMART\nTASEP")

initial_state[initial_state != 0] = 1

# invert around y axis
# initial_state = initial_state[:, ::-1]

envParams = EnvParams(render_mode="human",
                      length=100,
                      width=100,
                      window_height=750,
                      initial_state=initial_state,
                      distinguishable_particles=True,
                      use_speeds=True,
                      allow_wait=True,
                      sigma=10,
                      observation_distance=3,
                      moves_per_timestep=30)

hyperparams = Hyperparams(BATCH_SIZE=32,
                          GAMMA=0.85,
                          EPS_START=0.9,
                          EPS_END=0.05,
                          EPS_DECAY=100_000,
                          TAU=0.005,
                          LR=0.001,
                          MEMORY_SIZE=500_000)

trainer = Trainer.load(41, env_params=envParams, wait_initial=10, do_plot=False)

trainer.run(reset_stats=True)
