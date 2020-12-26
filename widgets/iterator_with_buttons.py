"""
Iterate + apply a function, and at each such step display widget buttons to get input from a user.

The MIT License (MIT)
Originally created at 9/26/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""


import pandas as pd
import numpy as np
import IPython
from IPython.core.display import display
from ipywidgets import widgets, IntProgress


class ButtonIterator:
    def __init__(self, iterable, button_names, apply_function, button_layout=None):
        """ Iterate an iterable, apply a function at each step and collect button clicked for a collection of buttons.
        :param iterable: e.g., pandas dataframe, list, etc.
        :param buttons_name: list of strings, showing the display-name (and content) of the shown buttons
        :param apply_function: function to apply at each step

        Example of usage (to annotate if 10 numbers are even or odd manually):
            bti = ButtonIterator(range(10), button_names=['odd', 'even'], apply_function=lambda x: print(x))
            bti.start()
            print(bti.results)
        """

        self.iterable = iterable
        self.is_pandas = False
        if isinstance(self.iterable, pd.Series) or isinstance(self.iterable, pd.DataFrame):
            self.is_pandas = True # use it to take next element via .iloc()

        self.results = [] # store the results as you proceed
        self.apply_function = apply_function

        # Create progress bar
        self.progress_bar = IntProgress(min=0, max=len(self.iterable), value=1)
        self.progress_bar.bar_style = 'info'
        self.interface_container = [self.progress_bar]

        if button_layout is None:
            self.button_layout = widgets.Layout(width='30%', height='40px')

        # Create the buttons and assign actions
        for button_name in button_names:
            # Create the button
            button = widgets.Button(description=button_name, layout=self.button_layout)
            # Assign action method on each button
            button.on_click(self.log_selected_button)
            self.interface_container.append(button)

        # Create extra a "back" & "next" buttons to change the iterators position manually
        back_button = widgets.Button(description='|Back|', layout=self.button_layout)
        next_button = widgets.Button(description='|Next|', layout=self.button_layout)
        back_button.on_click(self._back_action)
        next_button.on_click(self._next_action)
        self.interface_container.extend([back_button, next_button])

        # Create the GUI interface
        self.interface = widgets.VBox(self.interface_container)

        # Current index in the iterator
        self.current_index =  0

    def _move_current_index(self, val):
        self.current_index += val

        if self.current_index < 0:  # too many backs.
            self.current_index = len(self.iterable) - 1

    def _back_action(self, _):
        self.results.pop()
        self._move_current_index(-1)
        self.update_interface()

    def _next_action(self, _):
        self.results.append(None) # skipping
        self._move_current_index(1)
        self.update_interface()

    def update_interface(self):
        # Clear the output
        IPython.display.clear_output()

        # Adjust the progress bar
        print(self.current_index, '/', len(self.iterable))

        # if self.results:
        #     print(self.iterable.iloc[self.current_index]['emotion'])

        self.progress_bar.value = self.current_index

        if self.current_index < len(self.iterable):
            self.progress_bar.bar_style = 'info'

            if self.is_pandas:
                current_row = self.iterable.iloc[self.current_index]
            else:
                current_row = self.iterable[self.current_index]

            self.apply_function(current_row)
            display(self.interface)
        else:
            self.progress_bar.bar_style = 'success'
            print('Done.')
            display(self.progress_bar)
            return self.results
            # raise StopIteration

    def log_selected_button(self, button):
        self.results.append(button.description)
        self.step()

    def step(self, first_time=False):
        # Forward the current index
        if not first_time:
            self._move_current_index(1)

        # Update the interface
        self.update_interface()

    def start(self):
        self.step(first_time=True)