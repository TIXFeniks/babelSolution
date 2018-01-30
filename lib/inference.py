"""
Implements ingraph sampling and beam search for seq2seq models
"""
import sys
import numpy as np
import tensorflow as tf
from tfnn.util import nested_map, is_scalar
from tfnn.ops.sliced_argmax import sliced_argmax
from lib.tensor_utils import infer_length, infer_mask
from collections import namedtuple
from warnings import warn


class GreedyInference:

    Stack = namedtuple("Stack", ['out_len', 'out_seq', 'logp', 'mask', 'dec_state', 'attnP', 'tracked'])

    def __init__(self, model, batch_placeholder, max_len=None, force_bos=True, force_eos=True,
                 get_tracked_outputs=lambda dec_state: [], crop_last_step=True,
                 back_prop=True, swap_memory=False, **flags):
        """
        Encode input sequence and iteratively decode output sequence.
        To be used in this fashion:
        trans = GreedyInference(model, {'inp':...}, ...)
        loss = cmon_do_something(trans.best_out, trans.best_scores)
        sess.run(loss)

        :type model: tfnn.task.seq2seq.inference.translate_model.TranslateModel
        :param batch: a dictionary that contains symbolic tensor {'inp': input token ids, shape [batch_size,time]}
        :param max_len: maximum length of output sequence, defaults to 2*inp_len + 3
        :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
        :param force_eos: if True, any token past initial EOS is guaranteed to be EOS
        :param get_tracked_outputs: callback that returns whatever tensor(s) you want to track on each time-step
        :param crop_last_step: if True, does not perform  additional decode __after__ last eos
                ensures all tensors have equal time axis
        :param back_prop: see tf.while_loop back_prop param
        :param swap_memory: see tf.while_loop swap_memory param
        :param **flags: you can add any amount of tags that encode and decode understands.
            e.g. greedy=True or is_train=True

        """
        self.batch_placeholder = batch_placeholder
        self.get_tracked_outputs = get_tracked_outputs

        inp_len = batch_placeholder.get('inp_len', infer_length(batch_placeholder['inp'], model.out_voc.eos))
        max_len = max_len if max_len is not None else (2 * inp_len + 3)

        first_stack = self.create_initial_stack(model, batch_placeholder, force_bos=force_bos, **flags)
        shape_invariants = nested_map(lambda v: tf.TensorShape([None for _ in v.shape]), first_stack)

        # Actual decoding
        def should_continue_translating(*stack):
            stack = self.Stack(*stack)
            return tf.reduce_any(tf.less(stack.out_len, max_len)) & tf.reduce_any(stack.mask)

        def inference_step(*stack):
            stack = self.Stack(*stack)
            return self.step(model, stack,  **flags)

        final_stack = tf.while_loop(
            cond=should_continue_translating,
            body=inference_step,
            loop_vars=first_stack,
            shape_invariants=shape_invariants,
            swap_memory=swap_memory,
            back_prop=back_prop,
        )

        _, outputs, scores, _, dec_states, attnP, tracked_outputs = final_stack
        if crop_last_step:
            attnP = attnP[:, :-1]
            tracked_outputs = nested_map(lambda out: out[:, :-1], tracked_outputs)

        if force_eos:
            out_mask = infer_mask(outputs, model.out_voc.eos)
            outputs = tf.where(out_mask, outputs, tf.fill(tf.shape(outputs), model.out_voc.eos))

        self.best_out = self.sample_out = outputs
        self.best_attnP = self.best_attnP = attnP
        self.best_scores = self.sample_scores = scores
        self.best_dec_states = self.dec_states = dec_states
        self.tracked_outputs = tracked_outputs

    def create_initial_stack(self, model, batch_placeholder, force_bos=True, **flags):
        inp = batch_placeholder['inp']
        batch_size = tf.shape(inp)[0]

        initial_state = model.encode(batch_placeholder, **flags)
        initial_attnP = model.get_attnP(initial_state)[:, None]
        initial_tracked = nested_map(lambda x: x[:, None], self.get_tracked_outputs(initial_state))

        if force_bos:
            initial_outputs = tf.cast(tf.fill((batch_size, 1), model.out_voc.bos), inp.dtype)
            initial_state = model.decode(initial_state, initial_outputs[:, 0], **flags)
            second_attnP = model.get_attnP(initial_state)[:, None]
            initial_attnP = tf.concat([initial_attnP, second_attnP], axis=1)
            initial_tracked = nested_map(lambda x, y: tf.concat([x, y[:, None]], axis=1),
                                         initial_tracked,
                                         self.get_tracked_outputs(initial_state),)
        else:
            initial_outputs = tf.zeros((batch_size, 0), dtype=inp.dtype)

        initial_logp = tf.zeros([batch_size], dtype='float32')
        initial_mask = tf.ones_like([batch_size], dtype='bool')
        initial_len = tf.shape(initial_outputs)[1]

        return self.Stack(initial_len, initial_outputs, initial_logp, initial_mask,
                          initial_state, initial_attnP, initial_tracked)

    def step(self, model, stack, **flags):
        """
        :type model: tfnn.task.seq2seq.inference.translate_model.TranslateModel
        :param stack: beam search stack
        :return: new beam search stack
        """
        out_len, out_seq, logp, mask, dec_states, attnP, tracked = stack

        # 1. sample
        batch_size = tf.shape(out_seq)[0]
        phony_slices = tf.range(batch_size)
        _, new_outputs, logp_next = model.sample(dec_states, logp, phony_slices, k=1, **flags)

        out_seq = tf.concat([out_seq, new_outputs], axis=1)
        logp = logp + logp_next[:, 0] * tf.cast(mask, 'float32')
        is_eos = tf.equal(new_outputs[:, 0], model.out_voc.eos)
        mask = tf.logical_and(mask, tf.logical_not(is_eos))

        # 2. decode
        new_states = model.decode(dec_states, new_outputs[:, 0], **flags)
        attnP = tf.concat([attnP, model.get_attnP(new_states)[:, None]], axis=1)
        tracked = nested_map(lambda seq, new: tf.concat([seq, new[:, None]], axis=1),
                             tracked, self.get_tracked_outputs(new_states)
                             )
        return self.Stack(out_len + 1, out_seq, logp, mask, new_states, attnP, tracked)

    def translate_batch(self, batch_data, **optional_feed):
        """
        Translates NUMERIC batch of data
        :param batch_data: dict {'inp':np.array int32[batch,time]}
        :optional_feed: any additional values to be fed into graph. e.g. if you used placeholder for max_len at __init__
        :return: best hypotheses' outputs and attnP
        """
        feed_dict = {placeholder: batch_data[k] for k, placeholder in self.batch_placeholder.items()}
        for k, v in optional_feed.items():
            feed_dict[k] = v

        out_ids, attnP = tf.get_default_session().run(
            [self.best_out, self.best_attnP],
            feed_dict=feed_dict)

        return out_ids, tf.transpose(attnP, [0, 2, 1])


class BeamSearchInference:

    def __init__(self, model, batch_placeholder, min_len=None, max_len=None,
                 beam_size=12, beam_spread=3, force_bos=True, if_no_eos='last',
                 back_prop=True, swap_memory=False, **flags
                 ):
        """
        Performs ingraph beam search for given input sequences (inp)
        Supports penalizing, pruning against best score and best score in beam (via beam_spread)
        :param model: something that implements TranslateModel
        :param batch_placeholder: whatever model can .encode,
            by default should be {'inp': int32 matrix [batch_size x time]}
        :param min_length: minimum valid output length. None means min_len=inp_len // 4 - 1
        :param max_len: maximum hypothesis length to consider,
            float('inf') means unlimited, None means max_len=2*inp_len + 3,
        :param beam_size: maximum number of hypotheses that can pass from one beam search step to another.
            The rest is pruned.
        :param beam_spread: maximum difference in score between a hypothesis and current best hypothesis.
            Anything below that is pruned.
        :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
        :param if_no_eos: if 'last', will return unfinished hypos if there are no finished hypos by max_len
                          elif 'initial', returns empty hypothesis
        :param back_prop: see tf.while_loop back_prop param
        :param swap_memory: see tf.while_loop swap_memory param

        :param **flags: whatever else you want to feed into model. This will be passed to encode, decode, etc.
            is_train - if True (default), enables dropouts and similar training-only stuff
            sampling_strategy - if "random", samples hypotheses proportionally to softmax(logits)
                                  otherwise(default) - takes top K hypotheses
            sampling_temperature -  if sampling_strategy == "random",
                performs sampling ~ softmax(logits/sampling_temperature)

        """
        assert if_no_eos in ['last', 'initial']
        assert np.isfinite(beam_spread) or max_len != float('inf'), "Must set maximum length if beam_spread is infinite"
        # initialize fields
        self.batch_placeholder = batch_placeholder
        inp_len = batch_placeholder.get('inp_len', infer_length(batch_placeholder['inp'], model.out_voc.eos))
        self.min_len = min_len if min_len is not None else inp_len // 4 - 1
        self.max_len = max_len if max_len is not None else 2 * inp_len + 3
        self.beam_size, self.beam_spread = beam_size, beam_spread
        self.force_bos, self.if_no_eos = force_bos, if_no_eos

        # actual beam search
        first_stack = self.create_initial_stack(model, batch_placeholder, force_bos=force_bos, **flags)
        shape_invariants = nested_map(lambda v: tf.TensorShape([None for _ in v.shape]), first_stack)

        def should_continue_translating(*stack):
            stack = self.BeamSearchStack(*stack)
            should_continue = self.should_extend_hypo(model, stack)
            return tf.reduce_any(should_continue)

        def expand_hypos(*stack):
            stack = self.BeamSearchStack(*stack)
            return self.beam_search_step(model, stack, **flags)

        last_stack = tf.while_loop(
            cond=should_continue_translating,
            body=expand_hypos,
            loop_vars=first_stack,
            shape_invariants=shape_invariants,
            back_prop=back_prop,
            swap_memory=swap_memory,
        )

        # crop unnecessary EOSes that occur if no hypothesis is updated on several last steps
        actual_length = infer_length(last_stack.best_out, model.out_voc.eos)
        max_length = tf.reduce_max(actual_length)
        last_stack = last_stack._replace(best_out=last_stack.best_out[:, :max_length])

        self.best_attnP = last_stack.best_attnP
        self.best_out = last_stack.best_out
        self.best_scores = last_stack.best_scores
        self.best_raw_scores = last_stack.best_raw_scores
        self.best_state = last_stack.best_dec_state

    def translate_batch(self, batch_data, **optional_feed):
        """
        Translates NUMERIC batch of data
        :param batch_data: dict {'inp':np.array int32[batch,time]}
        :optional_feed: any additional values to be fed into graph. e.g. if you used placeholder for max_len at __init__
        :return: best hypotheses' outputs and attnP
        """
        feed_dict = {placeholder: batch_data[k] for k, placeholder in self.batch_placeholder.items()}
        for k, v in optional_feed.items():
            feed_dict[k] = v

        out_ids, attnP = tf.get_default_session().run(
            [self.best_out, self.best_attnP],
            feed_dict=feed_dict)

        return out_ids, tf.transpose(attnP, [0, 2, 1])

    BeamSearchStack = namedtuple('BeamSearchStack', [
        # per hypo values
        'out',  # [batch_size x beam_size, nout], int
        'scores',  # [batch_size x beam_size ]
        'raw_scores',  # [batch_size x beam_size ]
        'attnP',  # [batch_size x beam_size, ninp, nout+1]
        'attnP_sum',  # [batch_size x beam_size, ninp]
        'dec_state',  # TranslateModel DecState nested structure of [batch_size x beam_size, ...]

        # per beam values
        'slices',  # indices of first hypo for each sentence [batch_size ]
        'out_len',  # total (maximum) length of a stack [], int
        'best_out',  # [batch_size, nout], int, padded with EOS
        'best_scores',  # [batch_size]
        'best_raw_scores',  # [batch_size]
        'best_attnP',  # [batch_size, ninp, nout+1], padded with EOS
        'best_dec_state',  # TranslateModel DecState; nested structure of [batch_size, ...]
    ])

    def _set_stack_shapes(self, stack):
        """
        Set correct shapes for BeamSearchStack tensors
        """
        stack.out.set_shape([None, None])
        stack.scores.set_shape([None])
        stack.raw_scores.set_shape([None])
        stack.attnP.set_shape([None, None, None])
        stack.attnP_sum.set_shape([None, None])
        stack.out_len.set_shape([])
        stack.best_out.set_shape([None, None])
        stack.best_scores.set_shape([None])
        stack.best_raw_scores.set_shape([None])
        stack.best_attnP.set_shape([None, None, None])
        return stack

    def create_initial_stack(self, model, batch, **flags):
        """
        Creates initial stack for beam search by encoding inp and optionally forcing BOS as first output
        :type model: tfnn.task.seq2seq.inference.TranslateModel
        :param batch: model inputs - whatever model can eat for self.encode(batch,**tags)
        :param force_bos: if True, forces zero-th output to be model.out_voc.bos. Otherwise lets model decide.
        """

        dec_state = dec_state_0 = model.encode(batch, **flags)
        attnP_0 = model.get_attnP(dec_state_0)
        batch_size = tf.shape(attnP_0)[0]

        out_len = tf.constant(0, 'int32')
        out = tf.zeros(shape=(batch_size, 0), dtype=tf.int32)  # [batch_size, nout = 0]

        if self.force_bos:
            bos = tf.fill(value=model.out_voc.bos, dims=(batch_size,))
            dec_state = dec_state_1 = model.decode(dec_state_0, bos, **flags)
            attnP_1 = model.get_attnP(dec_state_1)
            attnP = tf.stack([attnP_0, attnP_1], axis=2)  # [batch_size, ninp, 2]
            out_len += 1
            out = tf.concat([out, bos[:, None]], axis=1)

        else:
            attnP = attnP_0[:, :, None]  # [batch_size, ninp, 1]

        slices = tf.range(0, batch_size)
        empty_out = tf.fill(value=model.out_voc.eos, dims=(batch_size, tf.shape(out)[1]))

        # Create stack.
        return self._set_stack_shapes(self.BeamSearchStack(
            out=out,
            scores=tf.zeros(shape=(batch_size,)),
            raw_scores=tf.zeros(shape=(batch_size,)),
            attnP=attnP,
            attnP_sum=tf.reduce_sum(attnP, axis=-1),
            dec_state=dec_state,
            slices=slices,
            out_len=out_len,
            best_out=empty_out,
            best_scores=tf.fill(value=-float('inf'), dims=(batch_size,)),
            best_raw_scores=tf.fill(value=-float('inf'), dims=(batch_size,)),
            best_attnP=attnP,
            best_dec_state=dec_state,
        ))

    def should_extend_hypo(self, model, stack):
        """
        Returns a bool vector for all hypotheses where True means hypo should be kept, 0 means it should be dropped.
        A hypothesis is dropped if it is either finished or pruned by beam_spread or by beam_size
        Note: this function assumes hypotheses for each input sample are sorted by scores(best first)!!!
        """

        # drop finished hypotheses
        should_keep = tf.logical_not(
            tf.reduce_any(tf.equal(stack.out, model.out_voc.eos), axis=-1))  # [batch_size x beam_size]

        n_hypos = tf.shape(stack.out)[0]
        batch_size = tf.shape(stack.best_out)[0]
        batch_indices = hypo_to_batch_index(n_hypos, stack.slices)

        # prune by length
        if self.max_len is not None:
            within_max_length = tf.less_equal(stack.out_len, self.max_len)

            # if we're given one max_len per each sentence, repeat it for each batch
            if not is_scalar(self.max_len):
                within_max_length = tf.gather(within_max_length, batch_indices)

            should_keep = tf.logical_and(
                should_keep,
                within_max_length,
            )

        # prune by beam spread
        if self.beam_spread is not None:
            best_scores_for_hypos = tf.gather(stack.best_scores, batch_indices)
            pruned_by_spread = tf.less(stack.scores + self.beam_spread, best_scores_for_hypos)
            best_raw_scores_for_hypos = tf.gather(stack.best_raw_scores, batch_indices)
            pruned_by_raw_spread = tf.less(stack.raw_scores + self.beam_spread, best_raw_scores_for_hypos)

            not_pruned = tf.logical_not(tf.logical_or(pruned_by_spread, pruned_by_raw_spread))

            should_keep = tf.logical_and(should_keep, not_pruned)

        # pruning anything exceeding beam_size
        if self.beam_size is not None:
            # This code will use a toy example to explain itself: slices=[0,2,5,5,8], n_hypos=10, beam_size=2
            # should_keep = [1,1,1,0,1,1,1,1,0,1] (two hypotheses have been pruned/finished)

            # 1. compute index of each surviving hypothesis globally over full batch,  [0,1,2,3,3,4,5,6,7,7]
            survived_hypo_id = tf.cumsum(tf.cast(should_keep, 'int32'), exclusive=True)
            # 2. compute number of surviving hypotheses for each batch sample, [2,2,3,1]
            survived_hypos_per_input = tf.bincount(batch_indices, weights=tf.cast(should_keep, 'int32'),
                                                   minlength=batch_size, maxlength=batch_size)
            # 3. compute the equivalent of slices for hypotheses excluding pruned: [0,2,4,4,7]
            slices_exc_pruned = tf.cumsum(survived_hypos_per_input, exclusive=True)
            # 4. compute index of surviving hypothesis within one sample (for each sample)
            # index of input sentence in batch:       inp0  /inp_1\  /inp_2\, /inp_3\
            # index of hypothesis within input:      [0, 1, 0, 1, 1, 0, 1, 2, 0, 0, 1]
            # 'e' = pruned earlier, 'x' - pruned now:         'e'         'x'   'e'
            beam_index = survived_hypo_id - tf.gather(slices_exc_pruned, batch_indices)

            # 5. prune hypotheses with index exceeding beam_size
            pruned_by_beam_size = tf.greater_equal(beam_index, self.beam_size)
            should_keep = tf.logical_and(should_keep, tf.logical_not(pruned_by_beam_size))

        return should_keep

    def beam_search_step(self, model, stack, **flags):
        """
        Performs one step of beam search decoding. Takes previous stack, returns next stack.
        :type model: tfnn.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        """

        # Prune
        #     - Against best completed hypo
        #     - Against best hypo in beam
        #     - EOS translations
        #     - Against beam size
        should_keep = self.should_extend_hypo(model, stack)

        hypo_indices = tf.where(should_keep)[:, 0]
        stack = self.shuffle_beam(model, stack, hypo_indices)

        # Compute penalties, if any
        base_scores = self.compute_base_scores(stack, **flags)

        # Get top-beam_size new hypotheses for each input.
        # Note: we assume sample returns hypo_indices from highest score to lowest, therefore hypotheses
        # are automatically sorted by score within each slice.
        hypo_indices, words, delta_raw_scores = model.sample(stack.dec_state, base_scores, stack.slices,
                                                             self.beam_size, **flags
                                                             )

        # hypo_indices, words and delta_raw_scores may contain -1/-1/-inf triples for non-available hypotheses.
        # This can only happen if for some input there were 0 surviving hypotheses OR beam_size > n_hypos*vocab_size
        # In either case, we want to prune such hypotheses
        valid_indices = tf.where(tf.not_equal(tf.reshape(hypo_indices, [-1]), -1))[:, 0]
        hypo_indices = tf.gather(tf.reshape(hypo_indices, [-1]), valid_indices)
        words = tf.gather(tf.reshape(words, [-1]), valid_indices)
        delta_raw_scores = tf.gather(tf.reshape(delta_raw_scores, [-1]), valid_indices)

        stack = self.shuffle_beam(model, stack, hypo_indices)
        dec_state = model.decode(stack.dec_state, words, **flags)
        step_attnP = model.get_attnP(dec_state)
        # step_attnP shape: [batch_size * beam_size, ninp]

        # collect stats for the next step
        attnP_sum = stack.attnP_sum + step_attnP
        attnP = tf.concat([stack.attnP, step_attnP[..., None]], axis=-1)
        out = tf.concat([stack.out, words[..., None]], axis=-1)
        out_len = stack.out_len + 1

        raw_scores = stack.raw_scores + delta_raw_scores
        stack = stack._replace(raw_scores=raw_scores)
        scores = self.compute_scores(stack, **flags)

        # Compute sample id for each hypo in stack
        n_hypos = tf.shape(stack.out)[0]
        batch_indices = hypo_to_batch_index(n_hypos, stack.slices)

        # Mark finished hypos
        finished = tf.equal(out[:, -1], model.out_voc.eos)

        if self.min_len is not None:
            below_min_length = tf.less(out_len, self.min_len)
            if not is_scalar(self.min_len):
                below_min_length = tf.gather(below_min_length, batch_indices)

            finished = tf.logical_and(finished, tf.logical_not(below_min_length))

        if self.if_no_eos == 'last':
            # No hypos finished with EOS, but len == max_len, consider unfinished hypos
            reached_max_length = tf.equal(out_len, self.max_len)
            if not is_scalar(self.max_len):
                reached_max_length = tf.gather(reached_max_length, batch_indices)

            have_best_out = tf.reduce_any(tf.not_equal(stack.best_out, model.out_voc.eos), 1)
            no_finished_alternatives = tf.gather(tf.logical_not(have_best_out), batch_indices)
            allow_unfinished_hypo = tf.logical_and(reached_max_length, no_finished_alternatives)

            finished = tf.logical_or(finished, allow_unfinished_hypo)

        # select best finished hypo for each input in batch (if any)
        finished_scores = tf.where(finished, scores, tf.fill(tf.shape(scores), -float('inf')))
        best_scores, best_indices = sliced_argmax(finished_scores[:, None], stack.slices, 1)
        best_scores, best_indices = best_scores[:, 0], stack.slices + best_indices[:, 0]
        best_indices = tf.clip_by_value(best_indices, 0, tf.shape(stack.out)[0])

        # take the better one of new best hypotheses or previously existing ones
        new_is_better = tf.greater(best_scores, stack.best_scores)
        best_scores = tf.where(new_is_better, best_scores, stack.best_scores)
        best_raw_scores = tf.where(new_is_better, tf.gather(raw_scores, best_indices), stack.best_raw_scores)

        batch_size = tf.shape(stack.best_out)[0]
        eos_pad = tf.fill(value=model.out_voc.eos, dims=(batch_size, 1))
        padded_best_out = tf.concat([stack.best_out, eos_pad], axis=1)
        best_out = tf.where(new_is_better, tf.gather(out, best_indices), padded_best_out)

        zero_attnP = tf.zeros_like(stack.best_attnP[:, :, :1])
        padded_best_attnP = tf.concat([stack.best_attnP, zero_attnP], axis=-1)
        best_attnP = tf.where(new_is_better, tf.gather(attnP, best_indices), padded_best_attnP)

        # if better translation is reached, update it's state too
        new_best_dec_state = model.shuffle(stack.dec_state, best_indices)
        best_dec_state = model.switch(new_is_better, new_best_dec_state, stack.best_dec_state)

        return self._set_stack_shapes(self.BeamSearchStack(
            out=out,
            scores=scores,
            raw_scores=raw_scores,
            attnP=attnP,
            attnP_sum=attnP_sum,
            slices=stack.slices,
            out_len=out_len,
            best_out=best_out,
            best_scores=best_scores,
            best_attnP=best_attnP,
            best_raw_scores=best_raw_scores,
            dec_state=dec_state,
            best_dec_state=best_dec_state,
        ))

    def compute_scores(self, model, stack, **flags):
        """
        Compute hypothesis scores given beam search stack. Applies any penalties necessary.
        For quick prototyping, you can store whatever penalties you need in stack.dec_state
        :type model: tfnn.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        :return: float32 vector (one score per hypo)
        """
        return stack.raw_scores

    def compute_base_scores(self, model, stack, **flags):
        """
        Compute hypothesis scores to be used as base_scores for model.sample.
        This is usually same as compute_scores but scaled to the magnitude of log-probabilities
        :type model: tfnn.task.seq2seq.inference.TranslateModel
        :type stack: BeamSearchDecoder.BeamSearchStack
        :return: float32 vector (one score per hypo)
        """
        return self.compute_scores(model, stack, **flags)

    def shuffle_beam(self, model, stack, flat_indices):
        """
        Selects hypotheses by index from entire BeamSearchStack
        Note: this method assumes that both stack and flat_indices are sorted by sample index
        (i.e. first are indices for input0 are, then indices for input1, then 2, ... then input[batch_size-1]
        """
        n_hypos = tf.shape(stack.out)[0]
        batch_size = tf.shape(stack.best_out)[0]

        # compute new slices:
        # step 1: get index of inptut sequence (in batch) for each hypothesis in flat_indices
        sample_ids_for_slices = tf.gather(hypo_to_batch_index(n_hypos, stack.slices), flat_indices)
        # step 2: compute how many hypos per flat_indices
        n_hypos_per_sample = tf.bincount(sample_ids_for_slices, minlength=batch_size, maxlength=batch_size)
        # step 3: infer slice start indices
        new_slices = tf.cumsum(n_hypos_per_sample, exclusive=True)

        # shuffle everything else
        return stack._replace(
            out=tf.gather(stack.out, flat_indices),
            scores=tf.gather(stack.scores, flat_indices),
            raw_scores=tf.gather(stack.raw_scores, flat_indices),
            attnP=tf.gather(stack.attnP, flat_indices),
            attnP_sum=tf.gather(stack.attnP_sum, flat_indices),
            dec_state=model.shuffle(stack.dec_state, flat_indices),
            slices=new_slices,
        )


class PenalizedBeamSearchDecoder(BeamSearchDecoder):
    """
    Performs ingraph beam search for given input sequences (inp)
    Implements length and coverage penalties
    """
    def compute_scores(self, stack, len_alpha=1, coverage_beta=0, **flags):
        """
        Computes scores after length and coverage penalty
        :param len_alpha: coefficient for length penalty, score / ( [5 + len(output_sequence)] / 6) ^ len_alpha
        :param coverage_beta: coefficient for coverage penalty (additive)
            coverage_beta * sum_i {log min(1.0, sum_j {attention_p[x_i,y_j] }  )}
        :return: float32 vector (one score per hypo)
        """
        if coverage_beta:
            warn("whenever coverage_beta !=0, this code works as in http://bit.ly/2ziK5a8,"
                 "which may or may not be correct depending on your definition.")

        scores = stack.raw_scores
        if len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), len_alpha)
            scores /= length_penalty
        if coverage_beta:
            times_translated = tf.minimum(stack.attnP_sum, 1)
            coverage_penalty = tf.reduce_sum(
                tf.log(times_translated + sys.float_info.epsilon),
                axis=-1) * coverage_beta
            scores += coverage_penalty
        return scores

    def compute_base_scores(self, stack, len_alpha=1, **flags):
        """
        Compute hypothesis scores to be used as base_scores for model.sample
        :return: float32 vector (one score per hypo)
        """
        scores = self.compute_scores(stack, len_alpha=len_alpha, **flags)
        if len_alpha:
            length_penalty = tf.pow((1. + tf.to_float(stack.out_len) / 6.), len_alpha)
            scores *= length_penalty
        return scores


def hypo_to_batch_index(n_hypos, slices):
    """
    Computes index in batch (input sequence index) for each hypothesis given slices.
    :param n_hypos: number of hypotheses (tf int scalar)
    :param slices: indices of first hypo for each input in batch

    It should guaranteed that
     - slices[0]==0 (first hypothesis starts at index 0), otherwise output[:slices[0]] will be -1
     - if batch[i] is terminated, then batch[i]==batch[i+1]
    """
    is_next_sent_at_t = tf.bincount(slices, minlength=n_hypos, maxlength=n_hypos)
    hypo_to_index = tf.cumsum(is_next_sent_at_t) - 1
    return hypo_to_index
