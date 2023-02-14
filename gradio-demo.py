import gradio as gr


def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text", css=".primary-button {background-color: cyan")

demo.launch()