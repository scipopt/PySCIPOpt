###############
Event Handlers
###############

For the following let us assume that a Model object is available, which is created as follows:

.. code-block:: python

  from pyscipopt import Model

  scip = Model()

.. contents:: Contents

SCIP Events
===========
SCIP provides a number of events that can be used to interact with the solver. These events would describe a change in the model state before or during the solving process; For example a variable/constraint were added or a new best solution is found.
The enum :code:`pyscipopt.SCIP_EVENTTYPE` provides a list of all available events.


What's an Event Handler?
==============
Event handlers are used to react to events that occur during the solving process.
They are registered with the solver and are called whenever an event occurs.
The event handler can then react to the event by performing some action.
For example, an event handler can be used to update the incumbent solution whenever a new best solution is found.


Adding Event Handlers with Callbacks
====================================

The easiest way to create an event handler is providing a callback function to the model using the :code:`Model.attachEventHandlerCallback` method.
The following is an example the prints the value of the objective function whenever a new best solution is found:

.. code-block:: python

  from pyscipopt import Model, SCIP_EVENTTYPE

  def print_obj_value(model, event):
      print("New best solution found with objective value: {}".format(model.getObjVal()))

  m = Model()
  m.attachEventHandlerCallback(print_obj_value, [SCIP_EVENTTYPE.BESTSOLFOUND])
  m.optimize()


The callback function should have the following signature: :code:`def callback(model, event)`.
The first argument is the model object and the second argument is the event that occurred.


Adding Event Handlers with Classes
==================================

If you need to store additional data in the event handler, you can create a custom event handler class that inherits from :code:`pyscipopt.Eventhdlr`.
and then include it in the model using the :code:`Model.includeEventHandler` method. The following is an example that stores the number of best solutions found:

.. code-block:: python

    from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE


    class BestSolCounter(Eventhdlr):
        def __init__(self, model):
            Eventhdlr.__init__(model)
            self.count = 0

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            self.count += 1
            print("!!!![@BestSolCounter] New best solution found. Total best solutions found: {}".format(self.count))


    m = Model()
    best_sol_counter = BestSolCounter(m)
    m.includeEventhdlr(best_sol_counter, "best_sol_event_handler", "Event handler that counts the number of best solutions found")
    m.optimize()
    assert best_sol_counter.count == 1

