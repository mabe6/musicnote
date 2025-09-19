from tkinter import *
from tkinter import messagebox, filedialog, scrolledtext
import pyautogui  # do pobierania pozycji myszy
import time
import keyboard
import json

# --------------------
# Music++ - GUI (wersja: tylko jeden capture na raz + Cancel Waiting)
# Zmiany:
# - podczas capture tylko jedna nuta moze byc w trybie oczekiwania (inne przyciski sa wylaczone)
# - dodany przycisk "Cancel Waiting" aby przerwac oczekiwanie (polling F6, mozna anulowac)
# - kazdy przycisk notki korzystajacy z F6 zachowuje sie tak samo (start_capture_for_note)
# Komentarze bez polskich znakow.
# --------------------





# main window
window = Tk()
window.geometry("700x520")
window.title("Music++")
window.config(background="#31363F")
#zmienne globalne
offset_x = 5   # przesuniecie w prawo
offset_y = 5   # przesuniecie w dol

# dla każdej nuty lista pozycji (może być kilka)
note_positions = {note: [] for note in NN}

# aktualnie wybrana nuta (domyślnie pierwsza)
current_note_index = 0


NN1 = DoubleVar(window,value=0.00)
NNN1 = DoubleVar(window,value=0.00) # zmienna do Spinboxa
NN2 = DoubleVar(window,value=0)
NN3 = DoubleVar(window,value=0)
NN4 = DoubleVar(window,value=0)
NN5 = DoubleVar(window,value=0)
NN6 = IntVar(window,value=0)
# Funkcje

def Start(count=5):
   if count > 0:
        console.configure(state='normal')
        console.insert('end', f'{count}\n')
        console.see('end')
        console.configure(state='disabled')
        window.after(1000, Start, count-1)  # po 1 sekundzie zmniejszamy count
   else:
        console.configure(state='normal')
        console.insert('end', "Start!\n")
        console.see('end')
        console.configure(state='disabled')

def ff():
      if messagebox.askokcancel("Start", "Start in 5 sec?"):
        Start(5)
     


# lista i nazw notek i gui z nutkami
NN = [
    "F#0", "G1", "G#2", "A3", "A#4", "B5",
    "C6", "C#7", "D8", "D#9", "E10", "F11",
    "F#12", "G13", "G#14", "A15", "A#16", "B17",
    "C18", "C#19", "D20", "D#21", "E22", "F23", "F#24"
]
for i, note in enumerate(NN):
    
    btn = Button(window, text=note, width=5, height=1)
    btn.place(x=offset_x + 48 * (i % 5), y=offset_y + 30 * (i // 5));
# gui stuff
Button(window, text='Load Notes', width=8, height=1).place(x=105,y=160)
Button(window, text='Save Notes', width=8, height=1).place(x=175,y=160)

Button(window, text='Top Left', width=10, height=1).place(x=5,y=250)
Button(window, text='Bottom Right', width=10, height=1).place(x=90,y=250)

Button(window, text='Load NBS', width=8, height=1).place(x=5,y=300)
Button(window, text='Show Grid', width=8, height=1).place(x=75,y=300)

Button(window, text='Load config', width=8, height=1).place(x=145,y=300)
Button(window, text='Save config', width=8, height=1).place(x=215,y=300)

check = Checkbutton(window, text="Always On", variable=NN6,command=lambda: window.attributes('-topmost', NN6.get()))
check.place(x=5, y=180)

Label(window, text="Skew X", font=("Arial", 11)).place(x=5, y=340)
Spinbox(window,textvariable=NN1, increment=0.01 ,from_=-0.10,to=0.10,width=8,font=("Arial", 10)).place(x=5,y=370)

Label(window, text="Skew Y", font=("Arial", 11)).place(x=80, y=340)
Spinbox(window,textvariable=NNN1, increment=0.01 ,from_=-0.10,to=0.10,width=8,font=("Arial", 10)).place(x=80,y=370)

Label(window, text="Cols:", font=("Arial", 9)).place(x=5, y=395)
Entry(window, textvariable=NN2, width=8, font=("Arial", 10)).place(x=5,y=420)

Label(window, text="Rows:", font=("Arial", 9)).place(x=80, y=395)
Entry(window, textvariable=NN3, width=8, font=("Arial", 10)).place(x=80,y=420)

Label(window, text="Dots:", font=("Arial", 9)).place(x=5, y=445)
Spinbox(window,textvariable=NN4, increment=1 ,from_=0,to=100,width=8,font=("Arial", 10)).place(x=5,y=470)

Label(window, text="Delay(ms):", font=("Arial", 9)).place(x=80, y=445)
Entry(window, textvariable=NN5, width=8, font=("Arial", 10)).place(x=80,y=470)

console = scrolledtext.ScrolledText(window, width=40, height=15,pady=10, state='disabled')
console.place(x=320,y=0)

Button(window,command=ff, text='Start', width=5, height=1).place(x=617,y=270)





window.mainloop()
