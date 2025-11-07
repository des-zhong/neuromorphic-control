from PIL import Image, ImageDraw

# Create a blank image (1024x1024 pixels)
width, height = 2048, 2048
scale = width/160
image = Image.new("RGB", (width, height), color=(50, 50, 50))  # Asphalt color

# Draw the dashed white line along the y-axis
draw = ImageDraw.Draw(image)
line_width = 2  # Slimmer line
dash_length = 10
spacing = 10
def recenter_draw(start_x, start_y, end_x, end_y,line_width):
    draw.line([width//2+start_x*scale, height//2-start_y*scale, width//2+end_x*scale, height//2-end_y*scale], fill="white", width=line_width)

draw.rectangle(
        [0, height // 2 - line_width // 2-60, width, height // 2 + line_width // 2-60],
        fill="white",
    )

# recenter_draw(-25, 0, -25, 25,1)
draw.rectangle(
        [0, height // 2 - line_width // 2+60, width, height // 2 + line_width // 2+60],
        fill="white",
    )

draw.rectangle(
    [ height // 2 - line_width // 2+60, 0,  height // 2 + line_width // 2+60, height],
    fill="white",
)
draw.rectangle(
    [ height // 2 - line_width // 2-60, 0,  height // 2 + line_width // 2-60, height],
    fill="white",
)
for x in range(0, width, dash_length + spacing):
    draw.rectangle(
        [x, height // 2 - line_width // 2, x + dash_length, height // 2 + line_width // 2],
        fill="white",
    )
for y in range(0, height, dash_length + spacing):
    draw.rectangle(
        [ height // 2 - line_width // 2, y,  height // 2 + line_width // 2, y + dash_length],
        fill="white",
    )

# Save the texture
image.save("textures/road_texture.png")