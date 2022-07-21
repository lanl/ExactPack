import importlib

import numpy
import matplotlib
matplotlib.use("WXAgg")
import matplotlib.pylab as plt
import wx

import exactpack


class MainWindow(wx.Frame):
    """The main application window for the ExactPack GUI."""
    
    solver = None

    def __init__(self, parent):

        super(MainWindow, self).__init__(parent, title="ExactPack")

        # Sizer to hold all the widgets
        sizer = wx.BoxSizer(wx.VERTICAL)        

        # Add a problem name menu
        sizer.Add(wx.StaticText(self, label="Problem Name:"))
        self.solver_name = wx.ListBox(self, wx.ID_ANY, choices=exactpack.discover_solvers())
        self.Bind(wx.EVT_LISTBOX, self.select_solver, self.solver_name)
        sizer.Add(self.solver_name, 0, wx.EXPAND)

        # Add the problem domain controls
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 1, 0)
        hsizer.Add(wx.StaticText(self, label="Solution Time:"))             # Add a validator?
        self.soln_time = wx.TextCtrl(self, wx.ID_ANY, value="1.0")
        hsizer.Add(self.soln_time, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 1, 0)
        hsizer.Add(wx.StaticText(self, label="Xmin:"))
        self.xmin = wx.TextCtrl(self, wx.ID_ANY, value="0.0")
        hsizer.Add(self.xmin, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 1, 0)
        hsizer.Add(wx.StaticText(self, label="Xmax:"))
        self.xmax = wx.TextCtrl(self, wx.ID_ANY, value="1.0")
        hsizer.Add(self.xmax, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 1, 0)
        hsizer.Add(wx.StaticText(self, label="Npts:"))
        self.npts = wx.TextCtrl(self, wx.ID_ANY, value="50")
        hsizer.Add(self.npts, 0, wx.EXPAND)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(hsizer, 1, 0)
        hsizer.Add(wx.StaticText(self, label="Parameters:"))
        self.params = wx.TextCtrl(self, wx.ID_ANY, value="")
        hsizer.Add(self.params, 0, wx.EXPAND)

        # Add the action controls
        button_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        sizer.Add(button_sizer, 1, wx.EXPAND)
                
        plot_button = wx.Button(self, wx.ID_ANY, "&Plot")
        self.Bind(wx.EVT_BUTTON, self.plotter, plot_button)
        button_sizer.Add(plot_button, 0, wx.ALL, 5)
        
        clear_button = wx.Button(self, wx.ID_ANY, "&Clear and Plot")
        self.Bind(wx.EVT_BUTTON, self.clear_and_plot, clear_button)
        button_sizer.Add(clear_button, 0, wx.ALL, 5)

        save_button = wx.Button(self, wx.ID_ANY, "&Save")
        self.Bind(wx.EVT_BUTTON, self.save, save_button)
        button_sizer.Add(save_button, 0, wx.ALL, 5)

        quit_button = wx.Button(self, wx.ID_ANY, "&Quit")
        self.Bind(wx.EVT_BUTTON, self.quit, quit_button)
        button_sizer.Add(quit_button, 0, wx.ALL, 5)

        self.SetSizerAndFit(sizer)
        self.Show(True)

    def select_solver(self, event):
        """A callback to respond to a pick from the solver menu."""

        name = self.solver_name.GetStringSelection()
        namelist = name.split('.')

        try:
            mod = importlib.import_module('.'.join(namelist[:-1]))
            self.solver = getattr(mod, namelist[-1])
        except:
            dialog = wx.MessageDialog(self, 
                                     "Can't find requested module {}".format(name),
                                     "Can't find module",
                                     style=wx.OK)
            dialog.ShowModal()
            dialog.Destroy()
            
            return

        self.params.SetValue(",".join(["{}={}".format(p, getattr(self.solver, p, "<missing>"))
                                       for p in self.solver.parameters.keys()]))

    def plotter(self, event):
        """A callback to overlay plot the current solution."""
        
        if self.solver:
            time = float(self.soln_time.GetValue())
            xmin = float(self.xmin.GetValue())
            xmax = float(self.xmax.GetValue())
            npts = int(self.npts.GetValue())
            kwargs = {}
            params = self.params.GetValue().split(",")
            if params!=[""]:
                for p in params:
                    key, val = p.split("=")
                    kwargs[key] = float(val)

            self.soln = self.solver(**kwargs)(numpy.linspace(xmin, xmax, npts), time)
            self.soln.plot_all()
            plt.show()

    def clear_and_plot(self, event):
        """A callback to clear the plot and plot the current solution."""
        
        plt.clf()
        self.plotter(event)

    def save(self, event):
        """A callback to save the most recently plotted solution."""

        save_dialog = wx.FileDialog(self, "Save data to CSV file", "", "", 
                                    "CSV files (*.csv)|*.csv", 
                                    wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if save_dialog.ShowModal() == wx.ID_CANCEL:
            return

        self.soln.dump(save_dialog.GetPath())
        
    def quit(self, event):
        """A callback to close the main window."""
        
        self.Close(True)
        

def main():
    """A GUI for ExactPack.

    Try it from the command line::

        epgui
    """
            
    app = wx.App(False)
    frame = MainWindow(None)
    app.MainLoop()


if __name__=='__main__':
    main()
