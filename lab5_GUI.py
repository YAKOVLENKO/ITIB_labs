from tkinter import *
import numpy as np
from testfiles.LAB5_YAKOVLENKO_IU8_61 import NeuroNet

class IWindow:
    def __init__(self):
        self.pixel_len = 40
        self.initUI()


    def initUI(self):
        self.root = Tk()
        self.root.title('ЛР7')
        self.root.minsize(780, 700)
        self.packFrames()
        self.startCanvas()
        self.startButton()
        self.packLabels()
        self.root.mainloop()

    def startCanvas(self):
        self.packCanvas()
        self.sample1.bind('<Button1-Motion>', self.drawCanvas)
        self.sample1.bind('<ButtonPress-1>', self.drawCanvas)
        self.sample2.bind('<Button1-Motion>', self.drawCanvas)
        self.sample2.bind('<ButtonPress-1>', self.drawCanvas)
        self.sample3.bind('<Button1-Motion>', self.drawCanvas)
        self.sample3.bind('<ButtonPress-1>', self.drawCanvas)
        self.reality_before.bind('<Button1-Motion>', self.drawCanvas)
        self.reality_before.bind('<ButtonPress-1>', self.drawCanvas)

    def startButton(self):
        self.packButtons()

        self.ButtonClear1_sample.bind('<Button-1>',
                                      lambda event, canvas=[self.sample1]: self.clearCanvas(event, canvas))
        self.ButtonClear2_sample.bind('<Button-1>',
                                      lambda event, canvas=[self.sample2]: self.clearCanvas(event, canvas))
        self.ButtonClear3_sample.bind('<Button-1>',
                                      lambda event, canvas=[self.sample3]: self.clearCanvas(event, canvas))
        self.ButtonClearE_sample.bind('<Button-1>',
                                      lambda event, canvas=[self.sample1,
                                                            self.sample2,
                                                            self.sample3]: self.clearCanvas(event, canvas))

        self.ButtonClear_reality.bind('<Button-1>',
                                      lambda event, canvas=[self.reality_before]: self.clearCanvas(event, canvas))

        self.ButtonDrawCurr_sample.bind('<ButtonPress-1>', self.pressButtonDrawCurr_sample)
        self.ButtonDrawCurr_sample.bind('<ButtonRelease-1>', self.releaseButtonDrawCurr_sample)

        self.ButtonFirstSample_reality.bind('<ButtonPress-1>',
                                            lambda event, sample=0:
                                            self.pressSetSample(event, sample))

        self.ButtonSecondSample_reality.bind('<ButtonPress-1>',
                                            lambda event, sample=1:
                                            self.pressSetSample(event, sample))

        self.ButtonThirdSample_reality.bind('<ButtonPress-1>',
                                            lambda event, sample=2:
                                            self.pressSetSample(event, sample))

        self.ButtonRecocnize_reality.bind('<ButtonPress-1>', self.pressRecognize)

    def packFrames(self):
        self.SampleFrame = LabelFrame(self.root, bd=2, text='', bg='pink')
        self.SampleFrame.place(x=10, y=10, height=380, width=750)

        self.RealityFrame = LabelFrame(self.root, bd=2, text='', bg='lightblue')
        self.RealityFrame.place(x=10, y=400, height=280, width=510)

        self.ResultFrame = LabelFrame(self.root, bd=2, text='', bg='lightgreen')
        self.ResultFrame.place(x=530, y=400, height=280, width=230)

    def packLabels(self):
        self.result_label = Label(self.ResultFrame, width=28, padx=2, height=2, pady=2, bg='white',
                                  text='Click "Recognize"', fg='lightgrey')
        self.result_label.place(x=10, y=225)

    # Создание и упаковка кнопок
    def packButtons(self):

        self.ButtonDraw_sample = Button(self.SampleFrame, text='Paint', width=8,
                                        height=13, bg='pink', activebackground='pink', relief='sunken',
                                        command=self.pressDraw_sample, state='disabled')
        # self.ButtonDraw_sample.place(x=640, y=10)

        self.ButtonClear1_sample = Button(self.SampleFrame, text='Clear', activebackground='pink',
                                          width=27, height=2, padx=4, bg='pink')
        self.ButtonClear1_sample.place(x=10, y=220)

        self.ButtonClear2_sample = Button(self.SampleFrame, text='Clear', activebackground='pink',
                                          width=27, height=2, padx=4, bg='pink')
        self.ButtonClear2_sample.place(x=220, y=220)

        self.ButtonClear3_sample = Button(self.SampleFrame, text='Clear', activebackground='pink',
                                          width=27, height=2, padx=4, bg='pink')
        self.ButtonClear3_sample.place(x=430, y=220)

        self.ButtonClearE_sample = Button(self.SampleFrame, text='Clear All', activebackground='pink',
                                          width=87, height=2, padx=4, bg='pink')
        self.ButtonClearE_sample.place(x=10, y=265)

        self.ButtonDrawCurr_sample = Button(self.SampleFrame, text='Show\nCurrent\nSamples', state='disabled',
                                            width=12, height=19, padx=2, bg='pink', activebackground='pink')
        self.ButtonDrawCurr_sample.place(x=640, y=10)

        self.ButtonRemember_sample = Button(self.SampleFrame, text='Remember',
                                            width=102, height=2, padx=2, command=self.pressRemember)
        self.ButtonRemember_sample.place(x=10, y=320)

        self.ButtonDraw_reality = Button(self.RealityFrame, text='Paint', relief='sunken',
                                         width=11, height=9, pady=6, bg='lightblue',
                                         activebackground='lightblue', command=self.pressDraw_reality)
        # self.ButtonDraw_reality.place(x=225, y=10)

        # self.ButtonErase_reality = Button(self.RealityFrame, text='Erase',
        #                                   width=11, height=9, pady=6, bg='lightblue',
        #                                  activebackground='lightblue', command=self.pressErase_reality)
        # self.ButtonErase_reality.place(x=315, y=10)
        #
        self.ButtonClear_reality = Button(self.RealityFrame, text='Clear', activebackground='lightblue',
                                          width=36, height=9, pady=6, padx = 3, bg='lightblue')
        self.ButtonClear_reality.place(x=226, y=10)

        self.ButtonRecocnize_reality = Button(self.RealityFrame, text='Recognize',
                                              width=67, height=2, padx=4, state='disabled')
        self.ButtonRecocnize_reality.place(x=10, y=225)

        self.ButtonFirstSample_reality = Button(self.RealityFrame, text='Set First\nSample', bg='lightblue',
                                              width=11, height=1, pady=7, activebackground='lightblue',
                                                state='disabled')

        self.ButtonFirstSample_reality.place(x=225, y=175)

        self.ButtonSecondSample_reality = Button(self.RealityFrame, text='Set Second\nSample', bg='lightblue',
                                                width=11, height=1, pady=7, activebackground='lightblue',
                                                 state='disabled')

        self.ButtonSecondSample_reality.place(x=315, y=175)

        self.ButtonThirdSample_reality = Button(self.RealityFrame, text='Set Third\nSample', bg='lightblue',
                                                 width=11, height=1, pady=7, activebackground='lightblue',
                                                state='disabled')

        self.ButtonThirdSample_reality.place(x=405, y=175)

    # Превращение полотна в вектор из 1 и -1
    def vectorizeCanvas(self, coordinates):

        black_pixels = list()
        vector = list()

        # for coordinates in self.getCanvasFilledPixels([canvas])[0]:
        for coordinate in coordinates:
            cell_number = coordinate[1] / self.pixel_len + coordinate[0] / self.pixel_len * \
                                                            (200 / self.pixel_len)
            if cell_number > -1 and cell_number < 25:
                black_pixels.append(cell_number)

        black_pixels = set(black_pixels)

        for pixel in range(0, 25):
            if pixel not in black_pixels:
                vector.append(-1)
            else:
                vector.append(1)

        return vector

    def clearCanvas(self, event, canvases):
        for canvas in canvases:
            canvas.delete('all')
            self.createCanvasGrid(canvas)

    def drawCanvas(self, event):
        coord_x, coord_y = event.x - event.x % self.pixel_len, event.y - event.y % self.pixel_len

        if event.widget == self.reality_before:
            button = self.ButtonDraw_reality
        else:
            button = self.ButtonDraw_sample

        if button['relief'] == 'sunken':
            event.widget.create_rectangle(coord_x, coord_y, coord_x + 40, coord_y + 40,
                                          fill='black', outline='lightgrey')

        else:
            event.widget.create_rectangle(coord_x, coord_y, coord_x + 40, coord_y + 40,
                                          fill='white', outline='lightgrey')

        self.root.update()

    def pressDraw_sample(self):
        self.ButtonDraw_sample.configure(relief = 'sunken')
        self.sample1.configure(cursor='pencil')
        self.sample2.configure(cursor='pencil')
        self.sample3.configure(cursor='pencil')

    # Получить координаты заполненных пикселей данных canvases
    def getCanvasFilledPixels(self, canvases):

        coordinates = list()

        for canvas in canvases:
            canvas_coordinates = list()
            for pixel in canvas.find_all()[200 * 2 // self.pixel_len :]:
                coords = canvas.coords(pixel)
                if coords not in canvas_coordinates:
                    canvas_coordinates.append(canvas.coords(pixel))

            # canvas_coordinates = set(canvas_coordinates)
            coordinates.append(canvas_coordinates)

        return coordinates

    def changeResultText(self, sample_vecrors, result_vector):
        for index, vector in enumerate(sample_vecrors):
            if vector == result_vector:
                self.result_label.configure(text='Образец ' + str(index + 1), fg='black')
                return
            self.result_label.configure(text='Химера', fg='black')


    def pressRecognize(self, event):
        samples_vector = [self.vectorizeCanvas(coordinates) for coordinates in self.remembered_samples]
        current_vector = self.vectorizeCanvas(self.getCanvasFilledPixels([self.reality_before])[0])
        neuro_net = NeuroNet(np.array(samples_vector))
        new_coordinates = neuro_net.find_image(current_vector)
        self.changeResultText(samples_vector, new_coordinates)
        new_coordinates = self.restoreCoordinates([new_coordinates])
        self.drawByCoordinates(new_coordinates, [self.reality_after], 'black', 'lightgrey')
        return

    # Функция клавиши ButtonRemember_sample (Remember)
    def pressRemember(self):

        self.remembered_samples = self.getCanvasFilledPixels([self.sample1,
                                                          self.sample2,
                                                          self.sample3])
        self.ButtonDrawCurr_sample.configure(state='normal')

        if self.ButtonRecocnize_reality['state'] == 'disabled':
            self.ButtonRecocnize_reality.configure(state='normal')
            self.ButtonFirstSample_reality.configure(state='normal')
            self.ButtonSecondSample_reality.configure(state='normal')
            self.ButtonThirdSample_reality.configure(state='normal')

    def pressSetSample(self, event, sample):
        self.drawByCoordinates([self.remembered_samples[sample]], [self.reality_before], 'black', 'lightgrey')

    def pressErase_sample(self):
        self.ButtonDraw_sample.configure(relief='raised')
        self.ButtonErase_sample.configure(relief='sunken')
        self.sample1.configure(cursor='tcross')
        self.sample2.configure(cursor='tcross')
        self.sample3.configure(cursor='tcross')

    # Восстановить координаты из векторов
    def restoreCoordinates(self, vectors):
        vectors_coordinates = list()
        for vector in vectors:
            current_vector_coordinates = list()
            for index, pixel in enumerate(vector):
                if pixel == 1:
                    x1_coordinate = int(index / (200 / self.pixel_len)) * self.pixel_len
                    x2_coordinate = x1_coordinate + self.pixel_len
                    y1_coordinate = (index % (200 / self.pixel_len)) * self.pixel_len
                    y2_coordinate = y1_coordinate + self.pixel_len
                    current_vector_coordinates.append([x1_coordinate, y1_coordinate,
                                                       x2_coordinate, y2_coordinate])
            vectors_coordinates.append(current_vector_coordinates)
            return vectors_coordinates

    # Нарисовать по координатам
    def drawByCoordinates(self, coordinates, canvases, fill_color, outline_color):
        for canvas_coordinates, canvas in zip(coordinates, canvases):
            canvas.delete('all')
            self.createCanvasGrid(canvas, outline_color)
            for coordinate in canvas_coordinates:
                canvas.create_rectangle(*coordinate, fill=fill_color, outline=outline_color)
        self.root.update()

    # Функция нажатой клавиши self.ButtonDrawCurr_sample (Show Current Samples)
    def pressButtonDrawCurr_sample(self, event):

        self.current_samples = self.getCanvasFilledPixels([self.sample1,
                                                          self.sample2,
                                                          self.sample3])

        self.drawByCoordinates(self.remembered_samples, [self.sample1, self.sample2, self.sample3],
                               'lightgrey', 'lightblue')

    # Функция отпускания клавиши self.ButtonDrawCurr_sample (Show Current Samples)
    def releaseButtonDrawCurr_sample(self, event):

        self.drawByCoordinates(self.current_samples, [self.sample1, self.sample2, self.sample3],
                               'black', 'lightgrey')

    def pressDraw_reality(self):
        self.ButtonErase_reality.configure(relief='raised')
        self.ButtonDraw_reality.configure(relief = 'sunken')
        self.reality_before.configure(cursor='pencil')

    def pressErase_reality(self):
        self.ButtonDraw_reality.configure(relief='raised')
        self.ButtonErase_reality.configure(relief='sunken')
        self.reality_before.configure(cursor='tcross')

    def createCanvasGrid(self, canvas, color='lightgrey'):
        for x in range(0, 200, self.pixel_len):
            canvas.create_line(x, 0, x, 200, fill=color)

        for y in range(0, 200, self.pixel_len):
            canvas.create_line(0, y, 200, y, fill=color)

    def packCanvas(self):

        self.sample1 = Canvas(self.SampleFrame, bg='white', width=199, height=199, cursor='pencil')
        self.createCanvasGrid(self.sample1)
        self.sample1.place(x=10, y=10)

        self.sample2 = Canvas(self.SampleFrame, bg='white', width=199, height=199, cursor='pencil')
        self.createCanvasGrid(self.sample2)
        self.sample2.place(x= 220, y=10)

        self.sample3 = Canvas(self.SampleFrame, bg='white', width=199, height=199, cursor='pencil')
        self.createCanvasGrid(self.sample3)
        self.sample3.place(x=430, y=10)

        self.reality_before = Canvas(self.RealityFrame, bg='white', width=199, height=199, cursor='pencil')
        self.createCanvasGrid(self.reality_before)
        self.reality_before.place(x=10, y=10)

        self.reality_after = Canvas(self.ResultFrame, bg='white', width=199, height=199)
        self.createCanvasGrid(self.reality_after)
        self.reality_after.place(x=10, y=10)
