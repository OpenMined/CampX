from . import things

def ascii_art_to_game(art,
                      what_lies_beneath,
                      sprites=None,
                      drapes = None,
                      backdrop=things.Backdrop,
                      update_schedule=None,
                      occlusion_in_layers=True):

    ""