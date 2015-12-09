# -*- coding: utf-8 -*-
"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
``(batch_size, sequence_length, num_inputs)``.   The CustomRecurrentLayer can
also support more than one "feature" dimension (e.g. using convolutional
connections), but for all other layers, dimensions trailing the third
dimension are flattened.

The following recurrent layers are implemented:

.. currentmodule:: lasagne.layers

.. autosummary::
    :nosignatures:

    CustomRecurrentLayer
    RecurrentLayer
    LSTMLayer
    GRULayer

For recurrent layers with gates we use a helper class to set up the parameters
in each gate:

.. autosummary::
    :nosignatures:

    Gate

Please refer to that class if you need to modify initial conditions of gates.

Recurrent layers and feed-forward layers can be combined in the same network
by using a few reshape operations; please refer to the example below.

Examples
--------
The following example demonstrates how recurrent layers can be easily mixed
with feed-forward layers using :class:`ReshapeLayer` and how to build a
network with variable batch size and number of time steps.

>>> from lasagne.layers import *
>>> num_inputs, num_units, num_classes = 10, 12, 5
>>> # By setting the first two dimensions as None, we are allowing them to vary
>>> # They correspond to batch size and sequence length, so we will be able to
>>> # feed in batches of varying size with sequences of varying length.
>>> l_inp = InputLayer((None, None, num_inputs))
>>> # We can retrieve symbolic references to the input variable's shape, which
>>> # we will later use in reshape layers.
>>> batchsize, seqlen, _ = l_inp.input_var.shape
>>> l_lstm = LSTMLayer(l_inp, num_units=num_units)
>>> # In order to connect a recurrent layer to a dense layer, we need to
>>> # flatten the first two dimensions (our "sample dimensions"); this will
>>> # cause each time step of each sequence to be processed independently
>>> l_shp = ReshapeLayer(l_lstm, (-1, num_units))
>>> l_dense = DenseLayer(l_shp, num_units=num_classes)
>>> # To reshape back to our original shape, we can use the symbolic shape
>>> # variables we retrieved above.
>>> l_out = ReshapeLayer(l_dense, (batchsize, seqlen, num_classes))
"""
import collections
import copy
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .. import init
from .. import nonlinearities
from .. import utils

from . import helper
from .base import MergeLayer, Layer
from .dense import DenseLayer
from .input import InputLayer
from .shape import FlattenLayer

__all__ = [
    "RecurrentContainerLayer",
    "DenseCell",
    "Gate",
    "LSTMCell",
    "GRUCell",
    "TeacherForcingCell",
    "SamplerCell",
    "WrapperCell",
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "LSTMLayer",
    "GRULayer"
]


# identity of data is defined by (name, id). id is an arbitrary object
# source_cell is a Cell (for Cell-Cell connections), or None if not part of DAG
# init_layer is a Lasagne Layer (if the value is initialized from one), or None
# type is one of
#   sequence:       inputs and outputs with a timestep dimension
#   output:         (initial values for) outputs
#   lateral:        (initial values for) connections from a Cell to itself
#   non_sequence:   inputs without a timestep dimension
class Connection(object):
    def __init__(self, name, id, shape, type='sequence',
                 source_cell=None, init_layer=None, **tags):
        self.name = name
        self.id = id                    # object
        self.source_cell = source_cell  # Cell or None
        self.init_layer = init_layer    # Layer or None
        # without n_batch, n_steps
        self.shape = tuple(shape) if shape is not None else None
        self.type = type
        self.tags = set(tag for tag, value in tags.items() if value)
        assert type in ('sequence', 'output', 'lateral', 'non_sequence')
        assert not (self.precomputable and self.full_column)

    @property
    def precomputable(self):
        return 'precomputable' in self.tags

    @property
    def full_column(self):
        return 'full_column' in self.tags

    @property
    def placeholder(self):
        return 'placeholder' in self.tags

    @property
    def repeat_batch(self):
        return 'repeat_batch' in self.tags

    def __hash__(self):
        return hash(self.name) + hash(self.id)

    def __eq__(self, other):
        return (self.name, self.id) == (other.name, other.id)

    def __repr__(self):
        tags = '{' + ', '.join('{}=True'.format(tag)
                               for tag in sorted(self.tags)) + '}'
        return '{}({}, {}, {}, {}, {}, {}, {})'.format(
            self.__class__.__name__,
            self.name,
            self.id,
            self.shape,
            self.type,
            self.source_cell,
            self.init_layer,
            tags)


class Expression(object):
    def __init__(self, var, timestep):
        assert timestep in ('no', 'preshuffle', 'full', 'slice', 'flat')
        self.var = var
        self.timestep = timestep

    def to_full(self, n_batch, n_steps):
        timestep = self.timestep
        var = self.var
        if self.timestep == 'preshuffle':
            var = var.dimshuffle(
                1, 0, *range(2, var.ndim))
            timestep = 'full'
        elif self.timestep == 'flat':
            shape = var.shape
            # This strange use of a generator in a tuple was because
            # shape[1:] was raising a Theano error
            trailing_dims = tuple(
                shape[n] for n in range(1, var.ndim))
            var = T.reshape(var, (n_steps, n_batch) + trailing_dims)
            timestep = 'full'
        return Expression(var, timestep)

    def to_flat(self, n_batch, n_steps):
        assert self.timestep != 'preshuffle'
        timestep = self.timestep
        var = self.var
        if self.timestep == 'full':
            shape = var.shape
            # This strange use of a generator in a tuple was because
            # shape[2:] was raising a Theano error
            trailing_dims = tuple(
                shape[n] for n in range(2, var.ndim))
            var = T.reshape(var, (n_steps * n_batch,) + trailing_dims)
            timestep = 'flat'
        return Expression(var, timestep)

    def to_output(self, only_return_final=False, backwards=False):
        assert self.timestep != 'flat'
        if self.timestep != 'full':
            return self
        var = self.var
        if only_return_final:
            var = var[-1]
            return Expression(var, 'slice')
        else:
            if backwards:
                var = var[::-1]
            # dimshuffle back to (n_batch, n_steps, ...)
            var = var.dimshuffle(
                1, 0, *range(2, var.ndim))
            timestep = 'preshuffle'
        return Expression(var, timestep)

    def repeat_batch(self, n_batch):
        assert self.timestep not in ('full', 'flat')
        var = T.extra_ops.repeat(self.var, n_batch, axis=0)
        return Expression(var, self.timestep)

    def __repr__(self):
        name = self.var.name if self.var.name else 'unnamed'
        return '{}<{}, {}d, {}>'.format(
            self.__class__.__name__,
            name,
            self.var.ndim,
            self.timestep)


def make_expression(when, connection, var):
    if connection.type == 'non_sequence':
        # non-sequences never have a timestep
        timestep = 'no'
    elif when == 'initial':
        if connection.type == 'sequence':
            # input sequences provided as (n_batch, n_steps, ...)
            timestep = 'preshuffle'
        else:
            # inits provided as only the first timestep
            timestep = 'slice'
    elif when == 'scan':
        timestep = 'slice'
    elif when == 'after':
        timestep = 'full'
    else:
        raise RuntimeError(
            'Unrecognized when "{}" in make_expression'.format(when))
    return Expression(var, timestep)


class RecurrentContainerLayer(MergeLayer):
    def __init__(self,
                 n_batch, n_steps,
                 sequences={},
                 output_inits={},
                 mask_input=None,
                 gradient_steps=-1,
                 backwards=False,
                 unroll_scan=False,
                 precompute_input=True,
                 postcompute_output=True,
                 only_return_final=False,
                 name='RecurrentContainer'):
        # can't do super
        self.name = name
        self.params = collections.OrderedDict()

        if n_batch is None or n_steps is None:
            raise ValueError(
                'To use variable shapes for batch size or '
                'number of time steps, you must specify the Layer '
                'from which to get the size.')
        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        self.n_batch = n_batch
        self.n_steps = n_steps
        self.gradient_steps = gradient_steps
        self.backwards = backwards
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.postcompute_output = postcompute_output
        self.only_return_final = only_return_final

        # {name -> Connection}
        self.connections = collections.OrderedDict()  # order is important
        self.output_mapping = {}
        self.last_cells = set()

        # {Connection -> Expression}
        self.expressions = {}

        # [Connection]
        self.dummy_inits = []

        for (name, layer_or_array) in sequences.items():
            self._add(name, layer_or_array,
                      type='sequence', precomputable=True)
        for (name, layer_or_array) in output_inits.items():
            self._add(name, layer_or_array,
                      type='output', full_column=True, placeholder=True)
        if mask_input is not None:
            self.mask_in = self._add(
                '_mask', mask_input, type='sequence', precomputable=True)
        else:
            self.mask_in = None

    def _add(self, name, layer_or_array, type, **kwargs):
        assert name not in self.connections
        expression = None
        if isinstance(layer_or_array, Layer):
            init_layer, shape = layer_or_array, layer_or_array.output_shape
        elif isinstance(layer_or_array, np.ndarray):
            init_layer, shape = None, layer_or_array.shape
            # lift to theano
            expression = utils.create_param(layer_or_array, shape, name=name)
            if expression.name is None:
                expression.name = name
        else:
            raise RuntimeError(
                "{} must be Layer or numpy array. Theano shared variables "
                "and shapes are not supported. ".format(name))

        if isinstance(self.n_batch, int):
            assert shape[0] == self.n_batch
        if type == 'sequence':
            if isinstance(self.n_steps, int):
                assert shape[1] == self.n_steps
            # n_batch, n_steps removed from connection shape
            shape = shape[2:]
            timestep = 'preshuffle'
        elif type == 'output':
            timestep = 'slice'
        else:
            timestep = 'no'

        connection = Connection(name, self, shape, type,
                                init_layer=init_layer, **kwargs)
        self.connections[name] = connection
        if expression is not None:
            self.expressions[connection] = Expression(expression, timestep)
        return connection

    def filter_connections(self, **kwargs):
        # see filter_connection_list for kwargs (don't give the flags!)
        return filter_connection_list(self.connections.values(), **kwargs)

    def get(self, name):
        """Returns the named Connection."""
        if name not in self.connections:
            raise RuntimeError(
                'No Connection named "{}". '
                'You can only get Connections representing input sequences '
                'given in the __init__ variable "sequences", '
                'or full-column recurrent Connections '
                'given in the __init__ variable "output_inits".'.format(
                    name))
        return self.connections[name]

    def connect_outputs(self, connections):
        # connections is a dict (name -> conn)
        full_columns = self.filter_connections(full_column=True)
        full_column_keys = set([connection.name
                                for connection in full_columns])
        missing = full_column_keys - set(connections.keys())
        if len(missing) > 0:
            raise RuntimeError(
                "All full column recurrent connections "
                "must be connected with connect_outputs. "
                "Missing keys: {}".format(', '.join(missing)))

        mapping = []
        for (name, real) in connections.items():
            if real.source_cell is None:
                raise RuntimeError(
                    'Outputs must be outgoing connections from Cells, '
                    'not "{}"'.format(real))
            self.last_cells.add(real.source_cell)
            # copy and modify to prevent DAG from becoming circular
            real = copy.copy(real)
            real.type = 'output'
            real.source_cell = None
            real.tags.add('full_column')
            real.tags.discard('placeholder')
            # for the time slices, type should be 'output' or 'lateral'
            if name in self.connections:
                placeholder = self.connections[name]
                if not placeholder.placeholder:
                    raise RuntimeError(
                        'Connection {} is not an output.'.format(placeholder))
                mapping.append((placeholder, real))
            else:
                if real.type not in ('output', 'lateral'):
                    raise RuntimeError(
                        "Cannot use connection of type {} as a side output. "
                        "You need to pass side_output=True "
                        "to Cell.get_output".format(real.type))
                self.dummy_inits.append(real)
                # side output: scan requires a dummy initial value
                real_shape = tuple(real.shape)
                if real_shape[0] is None:
                    real_shape = (1,) + real_shape[1:]
                    real.tags.add('repeat_batch')
                expression = np.zeros(real_shape)
                expression = utils.create_param(expression,
                                                real_shape,
                                                name=name)
                expression = T.shape_padleft(expression)
                # FIXME: do we need to mark dummyness in Expression?
                self.expressions[real] = Expression(expression, 'slice')
                # no need to map side outputs

        for cell in self.get_all_cells():
            cell.replace_placeholders(mapping)
        for (placeholder, real) in mapping:
            for (name, connection) in self.connections.items():
                if connection == placeholder:
                    self.connections[name] = real
            for (connection, expression) in self.expressions.items():
                if connection == placeholder:
                    del self.expressions[connection]
                    self.expressions[real] = expression

        self.output_mapping = connections

    def get_output_for(self, inputs, **kwargs):
        # connect input expressions with the associated Connection
        connections = self.filter_connections(init_from_layer=True)
        for cell in self.get_all_cells():
            connections.extend(cell.filter_connections(init_from_layer=True))
        expressions = {}
        n_batch = self.n_batch if isinstance(self.n_batch, int) else None
        n_steps = self.n_steps if isinstance(self.n_steps, int) else None
        for (connection, expression) in zip(connections, inputs):
            expressions[connection] = make_expression('initial',
                                                      connection,
                                                      expression)
            if n_batch is None and self.n_batch == connection.init_layer:
                n_batch = expression.shape[0]
            if n_steps is None and self.n_steps == connection.init_layer:
                n_steps = expression.shape[1]
        if n_batch is None or n_steps is None:
            raise RuntimeError(
                'Source layer for n_batch ({}) or n_steps ({}) '
                'not found'.format(n_batch, n_steps))
        # combine with expressions from parameters
        expressions.update(self.expressions)
        all_cells = self.get_all_cells(**kwargs)
        for cell in all_cells:
            expressions.update(cell.get_parameters())

        # if the expression needs to be repeated for n_batch, do so
        for (conn, expression) in expressions.items():
            if conn.repeat_batch:
                expressions[conn] = expression.repeat_batch(n_batch)

        # input seqs should be provided as (n_batch, n_steps, ...)
        # but scan requires the iterable dimension to be first
        # so we need to dimshuffle to (n_steps, n_batch, ...)
        expressions = {conn: expression.to_full(n_batch, n_steps)
                       for (conn, expression) in expressions.items()}

        assert all(not conn.placeholder for conn in expressions.keys())

        # ## retrieve all recurrent connections
        # output inits are in self
        recurrent_connections = self.filter_connections(type='output')
        # dummy inits for side outputs
        recurrent_connections.extend(self.dummy_inits)

        full_column_recurrence = False
        non_sequence_connections = []
        for cell in all_cells:
            if len(cell.filter_connections(type='output',
                                           full_column=True,
                                           tags={'dummy_input': False},
                                           **kwargs)):
                # some cell uses a full column recurrence as input
                # (filtering out dummy_input prevents false positives
                # for side outputs)
                full_column_recurrence = True
            for lateral in cell.filter_connections(type='lateral',
                                                   **kwargs):
                # duplicates can occur e.g. due to outputs
                if lateral not in recurrent_connections:
                    recurrent_connections.append(lateral)
            non_sequence_connections.extend(cell.filter_connections(
                type='non_sequence', **kwargs))

        # ## retrieve the cells, grouped by manner of computation
        # postcomputation cells, if possible
        if self.postcompute_output and not full_column_recurrence:
            post_cells, last_recs, post_left_border = \
                get_all_cells(
                    self.last_cells,
                    lambda cell: cell.potentially_postcomputable,
                    tags=None, **kwargs)
        else:
            post_cells, last_recs, post_left_border = \
                [], list(self.last_cells), []

        # recurrent computation cells
        if self.precompute_input and not full_column_recurrence:
            def criterion(cell):
                return not cell.precomputable
        else:
            criterion = None
        rec_cells, last_pres, rec_left_border = \
            get_all_cells(last_recs, criterion, tags=None, **kwargs)

        # precomputation cells, if possible
        if self.precompute_input and not full_column_recurrence:
            pre_cells, _, _ = get_all_cells(last_pres, tags=None, **kwargs)
        else:
            pre_cells = []

        # impose ordering on borders
        rec_left_border = [cell for cell in rec_cells
                           if cell in rec_left_border]
        post_left_border = [cell for cell in post_cells
                            if cell in post_left_border]

        # perform precomputation
        if self.precompute_input and not full_column_recurrence:
            expressions = propagate_expressions(pre_cells,
                                                expressions,
                                                n_steps, n_batch,
                                                precompute=True,
                                                **kwargs)

        # build scan arguments
        border_connections = []
        for border_cell in rec_left_border:
            border_connections.extend(
                border_cell.filter_connections(**kwargs))
        sequence_expressions = []
        init_expressions = []
        non_sequence_expressions = []
        argument_order = []
        output_order = []
        for connection in filter_connection_list(border_connections,
                                                 type='sequence'):
            sequence_expressions.append(expressions[connection].var)
            argument_order.append(connection)
        if self.mask_in:
            sequence_expressions.append(expressions[self.mask_in].var)
            argument_order.append(self.mask_in)
        for connection in filter_connection_list(recurrent_connections,
                                                 type='output'):
            init_expressions.append(expressions[connection].var)
            argument_order.append(connection)
            output_order.append(connection)
        for connection in filter_connection_list(recurrent_connections,
                                                 type='lateral'):
            init_expressions.append(expressions[connection].var)
            argument_order.append(connection)
            output_order.append(connection)
        for connection in non_sequence_connections:
            non_sequence_expressions.append(
                expressions[connection].var)
            argument_order.append(connection)

        # define the step function that propagates through recurrence DAG
        def step_fn(*args):
            for (connection, expression) in zip(argument_order, args):
                expressions[connection] = make_expression('scan',
                                                          connection,
                                                          expression)
            results = propagate_expressions(rec_cells,
                                            expressions,
                                            n_steps, n_batch,
                                            precompute=False,
                                            **kwargs)
            if self.mask_in:
                for conn in output_order:
                    padded_mask = expressions[self.mask_in].var
                    expr_ndim = results[conn].var.ndim
                    if padded_mask.ndim < expr_ndim:
                        padded_mask = T.shape_padright(
                            expressions[self.mask_in].var,
                            expr_ndim - padded_mask.ndim)
                    result = T.switch(
                        padded_mask,
                        results[conn].var,      # new value if true
                        expressions[conn].var)  # old value if false
                    results[conn] = Expression(result, 'slice')
            return [results[conn].var for conn in output_order]

        # apply scan
        if len(rec_cells) == 0:
            print('WARNING: found nothing to scan')
        else:
            if self.unroll_scan:
                # Explicitly unroll the recurrence instead of using scan
                scan_fn = utils.unroll_scan
            else:
                # Scan op iterates over first dimension of input and repeatedly
                # applies the step function
                scan_fn = theano.scan

            outputs, updates = scan_fn(
                fn=step_fn,
                sequences=sequence_expressions,
                outputs_info=init_expressions,
                non_sequences=non_sequence_expressions,
                go_backwards=self.backwards,
                n_steps=n_steps)
            if len(updates) > 0:
                # FIXME: ugly hack for getting updates to theano.function
                self._stashed_updates = updates
                print('WARNING: updates returned by scan was non-empty. '
                      'Use the ugly hack: self._stashed_updates')
            if outputs is None:
                outputs = []
            elif isinstance(outputs, theano.Variable):
                outputs = [outputs]
            for (connection, expression) in zip(output_order, outputs):
                expressions[connection] = make_expression('after',
                                                          connection,
                                                          expression)

        # perform postcomputation
        if self.postcompute_output and not full_column_recurrence:
            expressions = propagate_expressions(post_cells,
                                                expressions,
                                                n_steps, n_batch,
                                                postcompute=True,
                                                **kwargs)

        # dimshuffle back to (n_batch, n_time_steps, n_features)
        results = {}
        for (name, conn) in self.output_mapping.items():
            expression = expressions[conn].to_output(
                self.only_return_final,
                self.backwards)
            results[name] = expression.var

        return results

    def get_all_cells(self, **kwargs):
        all_cells, _, _ = get_all_cells(self.last_cells,
                                        tags=None, **kwargs)
        return all_cells

    def get_params(self, **tags):
        # filter out dummy inputs unless asked for
        tags['dummy_input'] = tags.get('dummy_input', False)
        expressions = dict(self.expressions)
        for cell in self.get_all_cells():
            expressions.update(cell.get_parameters())
        filtered = [expressions[key].var for key
                    in filter_connection_list(expressions.keys(), **tags)]
        return utils.collect_shared_vars(filtered)

    def get_n_batch(self, input_shapes=None):
        if isinstance(self.n_batch, int):
            return self.n_batch
        if input_shapes is None:
            return None
        connections = self.filter_connections(init_from_layer=True)
        for (connection, shape) in zip(connections, input_shapes):
            if self.n_batch == connection.init_layer:
                return shape[0]
        raise RuntimeError('Source layer for n_batch not found')

    def get_n_steps(self, input_shapes=None):
        if isinstance(self.n_steps, int):
            return self.n_steps
        if input_shapes is None:
            return None
        connections = self.filter_connections(init_from_layer=True)
        for (connection, shape) in zip(connections, input_shapes):
            if self.n_steps == connection.init_layer:
                return shape[1]
        raise RuntimeError('Source layer for n_steps not found')

    # ## For Lasagne compatibility
    #
    @property
    def input_layers(self):
        connections = self.filter_connections(init_from_layer=True)
        for cell in self.get_all_cells():
            connections.extend(cell.filter_connections(init_from_layer=True))
        return [connection.init_layer for connection in connections]

    @property
    def input_shapes(self):
        return [layer.output_shape for layer in self.input_layers]

    def get_output_layer(self, output_name):
        """ Returns a DictKeyLayer keyed for a particular output.
        This layer can be used as input for other Lasagne layers.
        """
        rcl = self

        class DictKeyLayer(Layer):
            def __init__(self, output_name):
                self.output_name = output_name
                self.connection = rcl.output_mapping[output_name]

                # can't do super
                self.input_layer = rcl
                self.input_shape = (None,) + self.connection.shape
                self.name = 'DictKeyLayer'
                self.params = collections.OrderedDict()

            @property
            def output_shape(self):
                return self.get_output_shape_for(
                    (rcl.get_n_batch(), rcl.get_n_steps()))

            def get_output_shape_for(self, input_shape):
                n_batch, n_steps = input_shape
                shape = self.connection.shape
                if rcl.only_return_final:
                    return (n_batch,) + shape
                else:
                    return (n_batch, n_steps) + shape

            def get_output_for(self, input, **kwargs):
                expression = input[self.output_name]
                expression.name = self.output_name
                return expression

        return DictKeyLayer(output_name)

    def get_output_shape_for(self, input_shapes):
        return (self.get_n_batch(input_shapes),
                self.get_n_steps(input_shapes))


class IncomingExpressionHelper(object):
    def __init__(self, incoming, expressions):
        self._incoming = incoming
        self._expressions = expressions

    def __getattr__(self, name):
        if name not in self._incoming:
            raise AttributeError(
                '"{}" not an incoming connection'.format(name))
        if self._incoming[name] not in self._expressions:
            raise AttributeError(
                '"{}" ({}) not in incoming expressions'.format(
                    name, self._incoming[name]))
        return self._expressions[self._incoming[name]].var


class Cell(object):
    def __init__(self, name, precomputable=False):
        self.name = name
        # {name -> Connection}
        self.incoming = {}
        # {name -> Connection}
        self.outgoing = {}
        # {Connection -> Expression}
        self.expressions = {}
        # Potentially precomputable, unless blocked by input
        self._pot_precomp = precomputable

    @property
    def precomputable(self):
        return self._pot_precomp and \
            all(incoming.precomputable for incoming
                in self.filter_connections(from_cell=True))

    @property
    def potentially_postcomputable(self):
        return self._pot_precomp

    def add(self, name, source, shape, type, **kwargs):
        """Used by subclasses to register incoming connections"""
        assert name not in self.incoming
        assert not (self._pot_precomp and type == 'lateral')
        if isinstance(source, Layer):
            shape = [axis for axis in shape if axis is not None]
            connection = Connection(name, self, shape, type,
                                    init_layer=source, **kwargs)
            self.incoming[name] = connection
        elif isinstance(source, Connection):
            if type == 'lateral' and \
                    source.source_cell is not None and \
                    source.source_cell != self:
                raise RuntimeError(
                    'Cannot initialize a lateral Connection "{}" '
                    'with the output of another Cell {}. '.format(
                        name, source.source_cell))
            shape = [axis for axis in shape if axis is not None]
            self.incoming[name] = source
            connection = source
        else:
            shape_with_batch = shape
            pad_for_batch = False
            if shape[0] is None:
                if (isinstance(source, np.ndarray) and
                        len(source.shape) == len(shape_with_batch)):
                    # the array already has the batch dimension
                    shape_with_batch = (source.shape[0],) + shape[1:]
                elif isinstance(source, theano.Variable):
                    # the variable already has the batch dimension
                    # but the batch dimension is uncheckable
                    shape_with_batch = (1,) + shape[1:]
                else:
                    # batch dimension of size 1, will be repeated later
                    kwargs['repeat_batch'] = True
                    pad_for_batch = True
                    shape_with_batch = shape[1:]
                shape = shape[1:]

            expression = utils.create_param(source,
                                            shape_with_batch,
                                            name=name)
            if pad_for_batch:
                expression = T.shape_padleft(expression)
            if expression.name is None:
                expression.name = name
            # parameters should be trainable and regularizable by default
            kwargs['trainable'] = kwargs.get('trainable', True)
            kwargs['regularizable'] = kwargs.get('regularizable', True)
            if type in ('output', 'lateral'):
                # recurrent outputs must have this Cell as source_cell
                # to be included in the DAG
                source_cell = self
            else:
                # parameters should not be included in the DAG
                source_cell = None
            connection = Connection(name, source, shape, type,
                                    source_cell=source_cell, **kwargs)
            self.incoming[name] = connection
            self.expressions[connection] = make_expression('initial',
                                                           connection,
                                                           expression)

        if type == 'lateral':
            if connection.placeholder:
                # create a proper output instead of passing on placeholder
                connection = Connection(name, self, shape, type,
                                        source_cell=self, **kwargs)
            else:
                # laterals are both incomings and outgoings
                # copy to avoid modifying passed in Connections
                connection = copy.copy(connection)
                connection.source_cell = self
                # filtering full_column is for incoming connections
                connection.tags.discard('full_column')
            self.outgoing[name] = connection

    def add_outgoing(self, name, shape, **kwargs):
        """Used by subclasses to register outgoing connections.
        Must be called after the calls to add.
        """
        assert name not in self.outgoing
        connection = Connection(name, self, shape, 'sequence',
                                source_cell=self,
                                precomputable=self.precomputable,
                                **kwargs)
        self.outgoing[name] = connection

    def replace_placeholders(self, mapping):
        """Replaces placeholders for full column recurrent connections."""
        for (placeholder, real) in mapping:
            for (name, connection) in self.incoming.items():
                if connection == placeholder:
                    self.incoming[name] = real

    def get(self, name, side_output=False):
        """Returns the named (outgoing) Connection."""
        connection = self.outgoing[name]
        if side_output and not connection.type == 'output':
            if not connection.type == 'sequence':
                raise RuntimeError(
                    "side_output is used for converting normal Cell-Cell "
                    "sequence connections into outputs")
            connection.type = 'output'
            connection.tags.add('dummy_input')
        return connection

    def expression(self, name):
        connection = self.incoming[name]
        return self.expressions[connection].var

    def filter_connections(self,
                           type=None,
                           from_cell=False,
                           init_from_layer=False,
                           tags=None,
                           **kwargs):
        # subclasses can use **kwargs to disable connections
        tags = tags if tags else {}
        return filter_connection_list(self.incoming.values(),
                                      type=type,
                                      from_cell=from_cell,
                                      init_from_layer=init_from_layer,
                                      **tags)

    def make_output(self, expressions, layerwise=False, **kwargs):
        """Helper for producing the output dict"""
        timestep = 'flat' if layerwise else 'slice'
        result = {}
        for name, expression in expressions.items():
            # outputs are always slices
            try:
                result[self.outgoing[name]] = Expression(expression,
                                                         timestep)
            except KeyError:
                raise RuntimeError(
                    'Cell {} has no output "{}"'.format(self, name))
        return result

    def get_parameters(self):
        return dict(self.expressions)

    def get_output_for(self, expressions, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return '{}<{} @ {}>'.format(
            self.__class__.__name__,
            self.name,
            hex(id(self)))


class DenseCell(Cell):
    def __init__(self, incoming, n_units,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.),
                 nonlinearity=None,
                 name='dense'):
        super(DenseCell, self).__init__(name, precomputable=True)
        self.nonlinearity = \
            nonlinearity if nonlinearity else nonlinearities.identity
        if len(incoming.shape) != 1:
            raise RuntimeError(
                'Input to DenseCell should be of shape '
                '(n_batch, n_features). '
                'Received shape {}'.format(incoming.shape))
        n_dims_in = incoming.shape[0]
        self.add('incoming', incoming,
                 incoming.shape, 'sequence')
        self.add('W', W, (n_dims_in, n_units), 'non_sequence')
        self.add('b', b, (n_units,), 'non_sequence', regularizable=False)

        self.add_outgoing('out', (n_units,))

    def get_output_for(self, expressions, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        result = T.dot(e.incoming, e.W) + e.b.dimshuffle('x', 0)
        result = self.nonlinearity(result)
        return self.make_output({'out': result}, **kwargs)


# Gate cannot use Connection inputs, because of stacking
class Gate(object):
    def __init__(self,
                 W_in=init.Normal(0.1),
                 W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid,
                 name=None):
        self.W_in_init = W_in
        self.W_hid_init = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell_init = W_cell
        self.b_init = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity
        self.name = name

    def init_weights(self, n_dims_in, n_units):
        name = '_to_{}gate'.format(self.name) \
            if self.name is not None else ''

        self.W_in = utils.create_param(self.W_in_init,
                                       (n_dims_in, n_units))
        if self.W_in.name is None:
            self.W_in.name = 'W_in' + name
        self.W_hid = utils.create_param(self.W_hid_init,
                                        (n_units, n_units))
        if self.W_hid.name is None:
            self.W_hid.name = 'W_hid' + name
        self.b = utils.create_param(self.b_init,
                                    (n_units, ))
        if self.b.name is None:
            self.b.name = 'b' + name
        if hasattr(self, 'W_cell_init'):
            self.W_cell = utils.create_param(self.W_cell_init,
                                             (n_units,))
            if self.W_cell.name is None:
                self.W_cell.name = 'W_cell' + name


class LSTMCell(Cell):
    def __init__(self,
                 incoming,
                 n_units,
                 ingate=Gate(name='in'),
                 forgetgate=Gate(name='forget'),
                 cell=Gate(W_cell=None,
                           nonlinearity=nonlinearities.tanh,
                           name='cell'),
                 outgate=Gate(name='out'),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 cell_init=init.Constant(0.),
                 learn_init=False,
                 peepholes=True,
                 grad_clipping=0,
                 name='LSTM'):
        super(LSTMCell, self).__init__(name, precomputable=False)
        n_dims_in = incoming.shape[0]
        self.n_units = n_units
        self.ingate = ingate
        self.forgetgate = forgetgate
        self.cellgate = cell
        self.outgate = outgate
        self.nonlinearity = nonlinearity if nonlinearity \
            else nonlinearities.identity
        self.learn_init = learn_init
        self.peepholes = peepholes
        self.grad_clipping = grad_clipping

        for gate in (ingate, forgetgate, cell, outgate):
            if gate is not None:
                gate.init_weights(n_dims_in, self.n_units)

        # stack weights for faster computation
        self.W_in_stacked = T.concatenate(
            [self.ingate.W_in, self.forgetgate.W_in,
             self.cellgate.W_in, self.outgate.W_in], axis=1)
        self.W_in_stacked.name = 'W_in_stacked'
        self.W_hid_stacked = T.concatenate(
            [self.ingate.W_hid, self.forgetgate.W_hid,
             self.cellgate.W_hid, self.outgate.W_hid], axis=1)
        self.W_hid_stacked.name = 'W_hid_stacked'
        self.add('W_hid_stacked',
                 self.W_hid_stacked,
                 (n_units, 4 * self.n_units),
                 'non_sequence')
        self.b_stacked = T.concatenate(
            [self.ingate.b, self.forgetgate.b,
             self.cellgate.b, self.outgate.b], axis=0)
        self.b_stacked.name = 'b_stacked'
        if self.peepholes:
            self.add('W_cell_to_ingate',
                     self.ingate.W_cell,
                     (self.n_units,),
                     'non_sequence')
            self.add('W_cell_to_forgetgate',
                     self.forgetgate.W_cell,
                     (self.n_units,),
                     'non_sequence')
            self.add('W_cell_to_outgate',
                     self.outgate.W_cell,
                     (self.n_units,),
                     'non_sequence')

        # for precomputability, wrap incoming in DenseCell
        precomputed = DenseCell(incoming,
                                4 * self.n_units,
                                self.W_in_stacked,
                                self.b_stacked).get('out')
        self.add('precomputed', precomputed, precomputed.shape, 'sequence')
        self.add('hid',
                 hid_init,
                 (None, self.n_units,),
                 'lateral',
                 trainable=self.learn_init,
                 regularizable=False)
        self.add('cell',
                 cell_init,
                 (None, self.n_units,),
                 'lateral',
                 trainable=self.learn_init,
                 regularizable=False)

    def get_output_for(self, expressions, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        input_preact = e.precomputed
        hid_previous = e.hid
        cell_previous = e.cell

        # a slicing function to extract the input to each gate
        def slice_w(x, n):
            return x[:, n * self.n_units:(n + 1) * self.n_units]

        gates_preact = input_preact + T.dot(hid_previous, self.W_hid_stacked)

        # Clip gradients
        if self.grad_clipping:
            gates_preact = theano.gradient.grad_clip(
                gates_preact, -self.grad_clipping, self.grad_clipping)

        # Extract the pre-activation gate values
        ingate_preact = slice_w(gates_preact, 0)
        forgetgate_preact = slice_w(gates_preact, 1)
        cell_input_preact = slice_w(gates_preact, 2)
        outgate_preact = slice_w(gates_preact, 3)

        if self.peepholes:
            # Add peephole connections
            ingate_preact += cell_previous * self.ingate.W_cell
            forgetgate_preact += cell_previous * self.forgetgate.W_cell

        # Apply nonlinearities
        ingate_act = self.ingate.nonlinearity(ingate_preact)
        forgetgate_act = self.forgetgate.nonlinearity(forgetgate_preact)
        cell_input_act = self.cellgate.nonlinearity(cell_input_preact)

        # Compute new cell value
        cell = forgetgate_act * cell_previous + ingate_act * cell_input_act

        if self.peepholes:
            outgate_preact += cell * self.outgate.W_cell
        outgate_act = self.outgate.nonlinearity(outgate_preact)

        # Compute new hidden unit activation
        hid = outgate_act * self.nonlinearity(cell)

        return self.make_output({'hid': hid, 'cell': cell}, **kwargs)


class GRUCell(Cell):
    def __init__(self,
                 incoming,
                 n_units,
                 resetgate=Gate(W_cell=None, name='reset'),
                 updategate=Gate(W_cell=None, name='update'),
                 candidategate=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh,
                                    name='candidate'),
                 nonlinearity=nonlinearities.tanh,
                 hid_init=init.Constant(0.),
                 learn_init=False,
                 grad_clipping=0,
                 name='GRU'):
        super(GRUCell, self).__init__(name, precomputable=False)
        n_dims_in = incoming.shape[0]
        self.n_units = n_units
        self.resetgate = resetgate
        self.updategate = updategate
        self.candidategate = candidategate
        self.nonlinearity = nonlinearity if nonlinearity \
            else nonlinearities.identity
        self.learn_init = learn_init
        self.grad_clipping = grad_clipping

        for gate in (resetgate, updategate, candidategate):
            if gate is not None:
                gate.init_weights(n_dims_in, self.n_units)

        # stack weights for faster computation
        self.W_in_stacked = T.concatenate(
            [self.resetgate.W_in, self.updategate.W_in,
             self.candidategate.W_in], axis=1)
        self.W_in_stacked.name = 'W_in_stacked'
        self.W_hid_stacked = T.concatenate(
            [self.resetgate.W_hid, self.updategate.W_hid,
             self.candidategate.W_hid], axis=1)
        self.W_hid_stacked.name = 'W_hid_stacked'
        self.add('W_hid_stacked',
                 self.W_hid_stacked,
                 (n_units, 3 * self.n_units),
                 'non_sequence')
        self.b_stacked = T.concatenate(
            [self.resetgate.b, self.updategate.b,
             self.candidategate.b], axis=0)
        self.b_stacked.name = 'b_stacked'

        # for precomputability, wrap incoming in DenseCell
        precomputed = DenseCell(incoming,
                                3 * self.n_units,
                                self.W_in_stacked,
                                self.b_stacked).get('out')
        self.add('precomputed', precomputed, precomputed.shape, 'sequence')

        self.add('hid',
                 hid_init,
                 (None, self.n_units,),
                 'lateral',
                 trainable=self.learn_init,
                 regularizable=False)

    def get_output_for(self, expressions, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        input_preact = e.precomputed
        hid_previous = e.hid

        # a slicing function to extract the input to each gate
        def slice_w(x, n):
            return x[:, n * self.n_units:(n + 1) * self.n_units]

        hid_preact = T.dot(hid_previous, self.W_hid_stacked)

        # Clip gradients
        if self.grad_clipping:
            input_preact = theano.gradient.grad_clip(
                input_preact, -self.grad_clipping, self.grad_clipping)
            hid_preact = theano.gradient.grad_clip(
                hid_preact, -self.grad_clipping, self.grad_clipping)

        # Extract the pre-activation gate values
        resetgate = slice_w(hid_preact, 0) + slice_w(input_preact, 0)
        updategate = slice_w(hid_preact, 1) + slice_w(input_preact, 1)

        # Apply nonlinearities
        resetgate = self.resetgate.nonlinearity(resetgate)
        updategate = self.updategate.nonlinearity(updategate)

        candidate_in_preact = slice_w(input_preact, 2)
        candidate_hid_preact = slice_w(hid_preact, 2)
        candidate_preact = \
            candidate_in_preact + resetgate * candidate_hid_preact
        if self.grad_clipping:
            candidate_preact = theano.gradient.grad_clip(
                candidate_preact, -self.grad_clipping, self.grad_clipping)
        candidate_act = self.candidategate.nonlinearity(candidate_preact)

        # Compute new hidden unit activation
        hid = (1 - updategate) * hid_previous + updategate * candidate_act

        return self.make_output({'hid': hid}, **kwargs)


class TeacherForcingCell(Cell):
    def __init__(self,
                 teacher_in,
                 prediction_in,
                 mask=None,
                 flag='teacher',
                 name='TeacherForcing'):
        # might be precomputable, unless the column recurrence is active
        super(TeacherForcingCell, self).__init__(name, precomputable=True)
        self.add('teacher', teacher_in,
                 teacher_in.shape, 'sequence',
                 **{flag: True})
        if not isinstance(prediction_in, Connection):
            # We don't assert this to be a placeholder,
            # because it also works with a transforming
            # feedforward cell in between
            raise RuntimeError(
                "prediction_in must be a full column recurrent output. "
                "(use RecurrentContainerLayer.get)")
        if self.incoming['teacher'].shape != prediction_in.shape:
            raise RuntimeError(
                "teacher_in {} and prediction_in {} must have "
                "the same shape.".format(
                    self.incoming['teacher'].shape,
                    prediction_in.shape))

        self.incoming['prediction'] = prediction_in
        if mask is not None:
            self.add('mask', mask, mask.shape, 'sequence', mask=True)
        else:
            self.mask_in = None
        self.flag = flag

        self.add_outgoing('out', teacher_in.shape)

    def filter_connections(self,
                           type=None,
                           from_cell=False,
                           init_from_layer=False,
                           tags=None,
                           **kwargs):
        # uses **kwargs to disable column recurrent connection
        omit = set()
        if self.flag in kwargs:
            if kwargs[self.flag] == 'full':
                # full teacher forcing: ignore prediction completely
                # disables full column recurrence
                omit.add('prediction')
            elif kwargs[self.flag] == 'no':
                # only use the predictions: ignore teacher
                omit.add('teacher')
            # else: use mask to select between the two (needs all three)

        connections = [conn for (name, conn) in self.incoming.items()
                       if name not in omit]
        tags = tags if tags else {}
        return filter_connection_list(connections,
                                      type=type,
                                      from_cell=from_cell,
                                      init_from_layer=init_from_layer,
                                      **tags)

    def get_output_for(self, expressions, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        if self.flag not in kwargs:
            raise RuntimeError(
                "TeacherForcingCell requires setting the flag "
                "'{}' to one of 'full', 'no' or 'mask'".format(self.flag))
        if kwargs[self.flag] == 'full':
            # full teacher forcing: ignore prediction completely
            result = e.teacher
        elif kwargs[self.flag] == 'no':
            # only use the predictions: ignore teacher
            result = e.prediction
        elif kwargs[self.flag] == 'mask':
            if self.mask_in is None:
                raise RuntimeError('mask was None')
            # use mask to select between the two inputs
            # true (1) selects teacher
            result = T.switch(e.mask, e.teacher, e.prediction)
        else:
            raise RuntimeError(
                "Unrecognized value for '{}', must be "
                "one of 'full', 'no' or 'mask'".format(self.flag))
        return self.make_output({'out': result}, **kwargs)


class SamplerCell(Cell):
    """Samples from a multinomial distribution according to the
    probabilities given as input.

    Input should be of shape (n_batch, n_classes),
    and each row should be a probability distribution.
    To get that, you might do something like:
    c_probs = DenseCell(
        incoming,
        n_units=vocab_size,
        W=lasagne.init.Normal(),
        nonlinearity=lasagne.nonlinearities.softmax)
    """

    def __init__(self,
                 probabilities,
                 random_seed=None,
                 name='sampler'):
        super(SamplerCell, self).__init__(name, precomputable=True)
        if len(probabilities.shape) != 1:
            raise RuntimeError(
                'Input to SamplerCell should be of shape '
                '(n_batch, n_classes), and each row should be '
                'a probability distribution. '
                'Received shape {}'.format(probabilities.shape))
        self.add('probabilities', probabilities,
                 probabilities.shape, 'sequence')
        if random_seed is None:
            # limits for random seed values set by algorithm
            M1 = 2147483647
            M2 = 2147462579
            # initialize theano RNG using numpy RNG
            random_seed = [
                np.random.randint(0, M1),
                np.random.randint(0, M1),
                np.random.randint(1, M1),
                np.random.randint(0, M2),
                np.random.randint(0, M2),
                np.random.randint(1, M2)]
        self.random = RandomStreams(random_seed)

        self.add_outgoing('out', (1,))

    def get_output_for(self, expressions, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        idx = self.random.multinomial(pvals=e.probabilities).argmax(1)
        idx.name = 'sampled_idx'
        return self.make_output({'out': idx}, **kwargs)


class WrapperCell(Cell):
    """Wraps feedforward lasagne Layers to allow using them as Cells.
    The Layers MUST be specified with shapes as input.
    """
    def __init__(self,
                 incoming,
                 input_to_hidden,
                 hidden_to_hidden=None,
                 hid_init=init.Constant(0.),
                 nonlinearity=None,
                 learn_init=False,
                 grad_clipping=0,
                 name='wrapper'):
        precomputable = hidden_to_hidden is None
        super(WrapperCell, self).__init__(name, precomputable=precomputable)
        self.add('incoming', incoming, incoming.shape, 'sequence')
        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.grad_clipping = grad_clipping
        self.nonlinearity = \
            nonlinearity if nonlinearity else nonlinearities.identity
        for (param, tags) in input_to_hidden.params.items():
            name = 'input_param.{}'.format(param.name)
            self._add_param(name, param, tags)
        if hidden_to_hidden is not None:
            if hidden_to_hidden.output_shape[1:] != \
                    input_to_hidden.output_shape[1:]:
                raise ValueError('input_to_hidden and hidden_to_hidden '
                                 'must have matching output feature shapes.')
            if input_to_hidden.output_shape[1:] != \
                    hidden_to_hidden.input_shape[1:]:
                raise ValueError(
                    'output feature shape of input_to_hidden'
                    'must match input feature shape of hidden_to_hidden.')
            self.add('out',
                     hid_init,
                     (None,) + hidden_to_hidden.output_shape[1:],
                     'lateral',
                     trainable=self.learn_init,
                     regularizable=False)
            for (param, tags) in hidden_to_hidden.params.items():
                name = 'hid_param.{}'.format(param.name)
                self._add_param(name, param, tags)
        else:
            self.add_outgoing('out', input_to_hidden.output_shape[1:])

    def get_output_for(self, expressions, layerwise=False, **kwargs):
        e = IncomingExpressionHelper(self.incoming, expressions)
        main_preact = self.input_to_hidden.get_output_for(e.incoming,
                                                          **kwargs)

        if self.hidden_to_hidden is not None:
            hid_previous = e.out
            hid_preact = self.hidden_to_hidden.get_output_for(
                hid_previous, **kwargs)
            main_preact += hid_preact

        # Clip gradients
        if self.grad_clipping:
            main_preact = theano.gradient.grad_clip(
                main_preact, -self.grad_clipping, self.grad_clipping)

        # Apply nonlinearities
        result = self.nonlinearity(main_preact)

        return self.make_output({'out': result}, **kwargs)

    def _add_param(self, name, param, tags):
        tags = {tag: True for tag in tags}
        # shape of tensor is unknowable, so we set it to None
        connection = Connection(name, param, None,
                                'non_sequence', **tags)
        self.incoming[name] = connection
        self.expressions[connection] = Expression(param, 'no')


def filter_connection_list(connections,
                           type=None,
                           full_column=False,
                           from_cell=False,
                           init_from_layer=False,
                           **tags):
    results = []
    pos = set(tag for tag, value in tags.items() if value)
    neg = set(tag for tag, value in tags.items() if not value)
    for connection in connections:
        if from_cell and not isinstance(connection.source_cell, Cell):
            continue
        if init_from_layer and not isinstance(connection.init_layer, Layer):
            continue
        if type is not None and connection.type != type:
            continue
        if full_column and not connection.full_column:
            continue
        if len(pos - connection.tags) > 0:
            continue    # some pos tag not present
        if len(neg & connection.tags) > 0:
            continue    # some neg tag present
        results.append(connection)
    return results


def get_all_cells(cells, criterion=None, tags=None, **kwargs):
    criterion = criterion if criterion else lambda cell: True
    tags = tags if tags else {}
    seen = set()
    done = set()
    result = []             # all cells meeting criterion
    right_border = set()    # rightmost cells failing criterion
    left_border = set()     # leftmost cells meeting criterion

    queue = collections.deque()
    for cell in cells:
        if criterion(cell):
            queue.append(cell)
        else:
            right_border.add(cell)

    while queue:
        # Peek at the leftmost node in the queue.
        cell = queue[0]
        if cell not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(cell)
            incomings = cell.filter_connections(from_cell=True,
                                                tags=tags, **kwargs)
            # Remove self-connections
            incomings = [inc for inc in incomings
                         if not inc.source_cell == cell]

            if len(incomings) == 0:
                # no parents: must be on left border
                left_border.add(cell)

            for incoming in incomings:
                if criterion(incoming.source_cell):
                    queue.appendleft(incoming.source_cell)
                else:
                    # source_cell fails crit
                    right_border.add(incoming.source_cell)
                    # cell has failing parent
                    left_border.add(cell)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the cell
            # from the queue and append it to the result list.
            queue.popleft()
            if cell not in done:
                result.append(cell)
                done.add(cell)
    return result, right_border, left_border


def propagate_expressions(cells,
                          expressions,
                          n_steps, n_batch,
                          precompute=False,
                          postcompute=False,
                          **kwargs):
    expressions = dict(expressions)
    layerwise = precompute or postcompute
    if layerwise:
        # flatten the timestep dimension and the minibatch dimension
        expressions = {conn: expression.to_flat(n_batch, n_steps)
                       for (conn, expression) in expressions.items()}

    for cell in cells:
        expressions.update(cell.get_output_for(expressions,
                                               layerwise=layerwise,
                                               **kwargs))

    if layerwise:
        # unflatten the timestep dimension and the minibatch dimension
        expressions = {conn: expression.to_full(n_batch, n_steps)
                       for (conn, expression) in expressions.items()}
    return expressions


# ## Reimplementations of old Lasagne recurrent API
#
class CustomRecurrentLayer(MergeLayer):
    """
    lasagne.layers.recurrent.CustomRecurrentLayer(incoming, input_to_hidden,
    hidden_to_hidden, nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False,
    learn_init=False, gradient_steps=-1, grad_clipping=0,
    unroll_scan=False, precompute_input=True, mask_input=None,
    only_return_final=False, **kwargs)

    A layer which implements a recurrent connection.

    This layer allows you to specify custom input-to-hidden and
    hidden-to-hidden connections by instantiating :class:`lasagne.layers.Layer`
    instances and passing them on initialization.  Note that these connections
    can consist of multiple layers chained together.  The output shape for the
    provided input-to-hidden and hidden-to-hidden connections must be the same.
    If you are looking for a standard, densely-connected recurrent layer,
    please see :class:`RecurrentLayer`.  The output is computed by

    .. math ::
        h_t = \sigma(f_i(x_t) + f_h(h_{t-1}))

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    input_to_hidden : :class:`lasagne.layers.Layer`
        :class:`lasagne.layers.Layer` instance which connects input to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which must end in a :class:`lasagne.layers.InputLayer` with the
        same input shape as `incoming`, except for the first dimension: When
        ``precompute_input == True`` (the default), it must be
        ``incoming.output_shape[0]*incoming.output_shape[1]`` or ``None``; when
        ``precompute_input == False``, it must be ``incoming.output_shape[0]``
        or ``None``.
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).  This layer may be connected to a chain of layers, which
        must end in a :class:`lasagne.layers.InputLayer` with the same input
        shape as `hidden_to_hidden`'s output shape.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    Examples
    --------

    The following example constructs a simple `CustomRecurrentLayer` which
    has dense input-to-hidden and hidden-to-hidden connections.

    >>> import lasagne
    >>> n_batch, n_steps, n_in = (2, 3, 4)
    >>> n_hid = 5
    >>> l_in = lasagne.layers.InputLayer((n_batch, n_steps, n_in))
    >>> l_in_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_in)), n_hid)
    >>> l_hid_hid = lasagne.layers.DenseLayer(
    ...     lasagne.layers.InputLayer((None, n_hid)), n_hid)
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(l_in, l_in_hid, l_hid_hid)

    The CustomRecurrentLayer can also support "convolutional recurrence", as is
    demonstrated below.

    >>> n_batch, n_steps, n_channels, width, height = (2, 3, 4, 5, 6)
    >>> n_out_filters = 7
    >>> filter_shape = (3, 3)
    >>> l_in = lasagne.layers.InputLayer(
    ...     (n_batch, n_steps, n_channels, width, height))
    >>> l_in_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer((None, n_channels, width, height)),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_hid_to_hid = lasagne.layers.Conv2DLayer(
    ...     lasagne.layers.InputLayer(l_in_to_hid.output_shape),
    ...     n_out_filters, filter_shape, pad='same')
    >>> l_rec = lasagne.layers.CustomRecurrentLayer(
    ...     l_in, l_in_to_hid, l_hid_to_hid)

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        n_batch = input_shape[0] if input_shape[0] is not None else incoming
        n_steps = input_shape[1] if input_shape[1] is not None else incoming
        self.l_rcl = RecurrentContainerLayer(
                 n_batch, n_steps,
                 sequences={'in': incoming},
                 mask_input=mask_input,
                 gradient_steps=gradient_steps,
                 backwards=backwards,
                 unroll_scan=unroll_scan,
                 precompute_input=precompute_input,
                 postcompute_output=True,
                 only_return_final=only_return_final)
        if precompute_input:
            if input_to_hidden.input_shape[0] is not None \
                    and input_to_hidden.input_shape[0] != n_batch * n_steps:
                raise ValueError(
                    'First dimension of input_to_hidden must be '
                    'n_batch * n_steps or None when precompute_input=True')
        elif hidden_to_hidden is not None:
            if input_to_hidden.input_shape[0] is not None \
                    and input_to_hidden.input_shape[0] != \
                    hidden_to_hidden.input_shape[0]:
                raise ValueError(
                    'First dimension of input_to_hidden must match '
                    'hidden_to_hidden or None when precompute_input=False')
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        self.c_wrapper = WrapperCell(
                 self.l_rcl.get('in'),
                 input_to_hidden,
                 hidden_to_hidden,
                 hid_init=hid_init,
                 learn_init=learn_init,
                 grad_clipping=grad_clipping)
        self.l_rcl.connect_outputs({'out': self.c_wrapper.get('out')})
        self.l_out = self.l_rcl.get_output_layer('out')
        super(CustomRecurrentLayer, self).__init__(self.l_rcl.input_layers)

    def get_params(self, **tags):
        return self.l_rcl.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        shape = self.l_rcl.get_output_shape_for(input_shapes)
        return self.l_out.get_output_shape_for(shape)

    def get_output_for(self, inputs, **kwargs):
        rcl_out = self.l_rcl.get_output_for(inputs, **kwargs)
        return self.l_out.get_output_for(rcl_out)

    # easy access to init
    @property
    def hid_init(self):
        return self.c_wrapper.expression('out')


class RecurrentLayer(CustomRecurrentLayer):
    """
    lasagne.layers.recurrent.RecurrentLayer(incoming, num_units,
    W_in_to_hid=lasagne.init.Uniform(), W_hid_to_hid=lasagne.init.Uniform(),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    Dense recurrent neural network (RNN) layer

    A "vanilla" RNN layer, which has dense input-to-hidden and
    hidden-to-hidden connections.  The output is computed as

    .. math ::
        h_t = \sigma(x_t W_x + h_{t-1} W_h + b)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    W_in_to_hid : Theano shared variable, numpy array or callable
        Initializer for input-to-hidden weight matrix (:math:`W_x`).
    W_hid_to_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).
    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
            incoming = InputLayer(incoming)
        else:
            input_shape = incoming.output_shape
        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        n_batch = input_shape[0]
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((n_batch, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None,
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, **kwargs)


class LSTMLayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial cell state (:math:`c_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` or
        `cell_init` are TensorVariables then the TensorVariable is used and
        `learn_init` is ignored for that initial state.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(name='in'),
                 forgetgate=Gate(name='forget'),
                 cell=Gate(W_cell=None,
                           nonlinearity=nonlinearities.tanh,
                           name='cell'),
                 outgate=Gate(name='out'),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        incoming = FlattenLayer(incoming, outdim=3)
        input_shape = incoming.output_shape
        n_batch = input_shape[0] if input_shape[0] is not None else incoming
        n_steps = input_shape[1] if input_shape[1] is not None else incoming
        if unroll_scan and input_shape[1] is None:
            raise ValueError(
                'n_steps can not be None when unroll_scan=True')
        self.l_rcl = RecurrentContainerLayer(
                 n_batch, n_steps,
                 sequences={'in': incoming},
                 mask_input=mask_input,
                 gradient_steps=gradient_steps,
                 backwards=backwards,
                 unroll_scan=unroll_scan,
                 precompute_input=precompute_input,
                 postcompute_output=True,
                 only_return_final=only_return_final)
        self.c_lstm = LSTMCell(
                 self.l_rcl.get('in'),
                 num_units,
                 ingate=ingate,
                 forgetgate=forgetgate,
                 cell=cell,
                 outgate=outgate,
                 nonlinearity=nonlinearity,
                 hid_init=hid_init,
                 cell_init=cell_init,
                 learn_init=learn_init,
                 peepholes=peepholes,
                 grad_clipping=grad_clipping)
        self.l_rcl.connect_outputs({'hid': self.c_lstm.get('hid')})
        self.l_out = self.l_rcl.get_output_layer('hid')
        super(LSTMLayer, self).__init__(self.l_rcl.input_layers)

    def get_params(self, **tags):
        return self.l_rcl.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        shape = self.l_rcl.get_output_shape_for(input_shapes)
        return self.l_out.get_output_shape_for(shape)

    def get_output_for(self, inputs, **kwargs):
        rcl_out = self.l_rcl.get_output_for(inputs, **kwargs)
        return self.l_out.get_output_for(rcl_out)

    # easy access to inits
    @property
    def hid_init(self):
        return self.c_lstm.expression('hid')

    @property
    def cell_init(self):
        return self.c_lstm.expression('cell')


class GRULayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    Gated Recurrent Unit (GRU) Layer

    Implements the recurrent step proposed in [1]_, which computes the output
    by

    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).

    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:

    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\

    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None, name='reset'),
                 updategate=Gate(W_cell=None, name='update'),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh,
                                    name='hidden_update'),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        incoming = FlattenLayer(incoming, outdim=3)
        input_shape = incoming.output_shape
        n_batch = input_shape[0] if input_shape[0] is not None else incoming
        n_steps = input_shape[1] if input_shape[1] is not None else incoming
        if unroll_scan and input_shape[1] is None:
            raise ValueError(
                'n_steps can not be None when unroll_scan=True')
        self.l_rcl = RecurrentContainerLayer(
                 n_batch, n_steps,
                 sequences={'in': incoming},
                 mask_input=mask_input,
                 gradient_steps=gradient_steps,
                 backwards=backwards,
                 unroll_scan=unroll_scan,
                 precompute_input=precompute_input,
                 postcompute_output=True,
                 only_return_final=only_return_final)
        self.c_gru = GRUCell(
                 self.l_rcl.get('in'),
                 num_units,
                 resetgate=resetgate,
                 updategate=updategate,
                 candidategate=hidden_update,
                 hid_init=hid_init,
                 learn_init=learn_init,
                 grad_clipping=grad_clipping)
        self.l_rcl.connect_outputs({'hid': self.c_gru.get('hid')})
        self.l_out = self.l_rcl.get_output_layer('hid')
        super(GRULayer, self).__init__(self.l_rcl.input_layers)

    def get_params(self, **tags):
        return self.l_rcl.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        shape = self.l_rcl.get_output_shape_for(input_shapes)
        return self.l_out.get_output_shape_for(shape)

    def get_output_for(self, inputs, **kwargs):
        rcl_out = self.l_rcl.get_output_for(inputs, **kwargs)
        return self.l_out.get_output_for(rcl_out)

    # easy access to init
    @property
    def hid_init(self):
        return self.c_gru.expression('hid')
