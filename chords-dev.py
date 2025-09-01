#! /usr/bin/env python

import math
import warnings
import copy
from itertools import ifilter, imap, islice, groupby, takewhile, product

import pyparsing 

"""
	Name conventions:
		mask = Tuple containing the tones that are to be in a chord
				e.g. all Major chords have mask (0, 4, 7)	
		ch_sf =  Tuple containing the fret to press for each string
				e.g. for D major (2, 3, 2, 0, 0, None)
		ch_fingers =  Tuple containg the finger allocation to the strings
				e.g. for D major (2, 3, 1, 0, 0, 0)

		together chstringfr & chfingers create a chord:
		class Chord
"""
class Chord:
	"""
		Joins the information of what to press on each string,
		with information of what fingers should do it.

		Provides pretty chord printing and some general chord
		information such as:
		self.minfret - the lowest fret pressed (determines position of hand)
		self.maxfret - the highest fret pressed
		self.basefret - the lowest fret on which something is played
		self.span - the width of chord one has to press (maxfret - minfret)

	"""
	def __init__(self, ch_sf, ch_fingers, num_fingers, instrument):
		"""
		parameters for __init__(self, ch_sf, ch_fingers, instrument):
			@ ch_sf - tuple of frets to press on each string
			@ ch_fingers - tuple with finger allocation
			@ num_fingers - number of fingers
			@ instrument - instance of Instrument class
		"""
		self.ch_sf = ch_sf
		self.string2fret = dict(ch_sf)

		self.ch_fingers = ch_fingers
		self.string2finger = dict(grip)

		self.num_fingers = num_fingers
		self.instrument = instrument

		self.basefret = min(self.filter_frets([None]))

		# minfret
		f = self.filter_frets([None,0])
		if len(f) > 0:
			self.minfret = min(f)
		else:
			self.minfret = 0
		# maxfret
		self.maxfret = max(self.filter_frets([None]))

		# chord span
		self.span = self.maxfret - self.minfret

		# initialize finger stats
		self._init_finger_stats()

	def __hash__(self):
		return hash((self.ch_sf, self.ch_fingers))

	def filter_frets(self, filterset):
		return map(lambda x: x[1], ifilter(lambda x: x[1] not in filterset, self.ch_sf))

	#def str_rich(self):
	#	l=[("string-fret",self.ch_sf), ("grip",self.ch_fingers)]
	#	return '\n'.join([tag + ": " + str(val) for tag, val in l] + ["chord:",str(self)])

	def _init_finger_stats(self):
			self.finger2fret = {}
			self.finger2strings = {}
			for string, finger in self.ch_fingers:
				if finger != None:
					self.finger2strings.setdefault(finger,[]).append(string)
					self.finger2fret[finger] = self.string2fret[string]

	def __str__(self):
		ch_sf, grip = self.ch_sf, self.grip
		str_fret, str_grip = self.string2fret, self.string2finger
		minfret,maxfret = self.minfret, self.maxfret
		longest_string_name = max( imap( lambda x : len(x), self.instrument.string_names) )
		lines=[]
		if minfret >= 3:
			lines.append(" "*(longest_string_name+3) + str(minfret))
		else:
			minfret = 1

		for num, string in enumerate(self.instrument.string_names):
			l = string + " "*(longest_string_name - len(string) +1) + "|"
			grp_str = "-"
			if str_grip[num] != None:
				grp_str = "%d"%(str_grip[num],)

			if str_fret[num] not in [None,0]:
				l += "---|"*(str_fret[num] - minfret) + "-%s-|"%(grp_str,) + "---|" * (maxfret-str_fret[num])
			else:
				l += "---|" * (maxfret - minfret + 1) + (" x" if str_fret[num] == None else ('' if minfret == 1 else ' o'))

			lines.append(l)
		return '\n'.join(lines)


class ManyToOneNamer:
	"""
		Provides a naming facility, something like a richer dict with
		forward (1 key -> 1 value) and backward ( 1 value -> some keys ) translation 
		
		Examples with the same result
			ManyToOneNamer(  {'#':1, 'is':1, 'b':-1, 'es':-1}  )
			ManyToOneNamer(  [('#',1), ('is',1), ('b',-1), ('es',-1)]  )
			ManyToOneNamer(  [  (['#','is'],1), (['b','es'],-1)  ]  )
			ManyToOneNamer(  [  (['#','is'],1), ('b',-1),('es',-1)]  )
	"""
	def __init__(self, inp):
		"""
		Param
			@inp - input: this can be either
				- dictionary which maps directly
					e.g. {'C':0, 'D':2, ...}
				- list of pairs [ p1, p2, .. ]
					if first element of some p is list
						than this is further unroled
		"""
		# dict for forward translation
		self.d = {}
		# dict for backward translation
		self.d_inv = {}
		if isinstance(inp, dict):
			self._init_drd(inp.items())
		elif isinstance(inp, list):
			self._init_drd(self._unroll_lazy(inp))
		else:
			raise RuntimeError

	def get_dicts(self):
		return copy.copy(self.d), copy.copy(self.d_inv)

	def _unroll_lazy(self, l):
		pairs=[]
		for f,s in l:
			if isinstance(f,list):
				for ff in f:
					pairs.append((ff,s))
			else:
				pairs.append((f,s))
		return pairs

	def _init_drd(self, pairs):
		d = self.d
		d_inv = self.d_inv
		for key, val in pairs:
			# one key should have ONLY one val
			if key in d and d[key] != val:
				raise RuntimeError("Error adding value '%s'. Key '%s' already has a value '%s'."%(str(val),str(key),str(d[key])))
			d[key] = val
			# make inverse dict, if one value has more keys, make list in
			d_inv.setdefault(val,[]).append(key)

class MusicSystem:
	"""
		Encapsulates all the information about music system used.
		That is:
			- what scalesize is used (how many equidistant "steps" in octave)
				e.g. 12
			- how tones are named and what are their values (a positive number of steps from 0)
				e.g. C=0 D=2 E=4 F=5 G=7 A=9 B=11
			- what tone modificators are used and their "shift"
				e.g. C alone is 0, C# is 1, so # has shift +1
			- what chords masks are used and what is their name
				e.g. major chord has mask (0, 4, 7), Dmajor is therefore (2, 6, 9)
		Handles conversions of
			names to tones and back
			names of chords to mask and back

		Provides the parser for tones and chords.
			tone name 	-> tone 		(e.g. F -> 5)
			tone 		-> tone name
			chord name	-> mask			(e.g. Dmaj -> (2,6,9,1))
	"""
	def __init__(self, scalesize, base_tones, tone_modificators, chord_masks):
		"""
			Params:
				scalesize - int declaring the number of equidistant units per octave
				base_tones - instance of ManyToOneNamer
				tone_modificators - instance of ManyToOneNamer 
				chord_masks - instance of ManyToOneNamer
		"""

		self.scalesize = scalesize
		# label -> tone, and vice versa
		self.l2t, self.t2l = base_tones.get_dicts()
		# modificator -> diff
		self.m2d, self.d2m = tone_modificators.get_dicts()
		# suffix -> mask
		self.chsuff2mask, self.mask2chsuff = chord_masks.get_dicts()

		self._generate_missing_tone_names()
		self._init_parsing()

	def tone_to_tone_name(self, tone):
		"""
			Converts a tone to the name of the tone.
			(e.g. tone_to_tone_name(0) = 'C' )
			(takes the first out of all possible different names)

			Raises KeyError if tone not nameable.
		"""
		ret = self.t2l.get(tone % self.scalesize, [])
		if len(ret):
			return ret[0]
		raise KeyError

	def tone_name_to_tone(self, label):
		"""
			Converts a tone name to the tone value.
			(e.g. tone_name_to_tone('Cis') = 1 )

			Raises KeyError if tone name does not name a tone.
		"""

		try:
			toks = self.p_tone.parseString(label)
		except pyparsing.ParseException:
			raise KeyError
		assert len(toks) == 1
		return toks[0]

	def mask_to_chordname(self, mask):
		"""
			Returns the name of chord described by mask.
			(takes the first out of all possible different names)

			Raises KeyError if mask is not nameable.
		"""
		# take first pair
		basetone, chmask = self._mask_to_base_n_chmask(mask).next()
		# name it
		tonename = self.tone_to_tone_name(basetone)
		chsuff = self.mask2chsuff[chmask]
		if len(chsuff):
			return tonename + chsuff[0]
		raise KeyError

	def chordname_to_mask(self, label):
		"""
			From a given name of the chord, returns its mask.

			Raises KeyError if chordname does not define a chord.
		"""
		try:
			toks = self.p_chord.parseString(label)
		except pyparsing.ParseException:
			raise KeyError(str(label))
		assert len(toks) == 1
		return toks[0]


	def _generate_missing_tone_names(self):
		def one_round(t2l, d2m):
			# maps 0  1 2  3 ..
			#   to 1 -1 2 -2
			def natural(n):
				return (-1)**(int(n)%2) * (1 + int(n) / 2)
			new = {}
			for tone in ifilter(lambda x: x not in t2l, xrange(self.scalesize)):
				for diff in imap(natural, xrange(self.scalesize)):
					try:
						l = [a + b for a,b in product(t2l[(tone - diff)%self.scalesize], d2m[diff]) ]
						new.setdefault(tone, []).extend(l)

						# when name found, break so we take the first one
						# (by ordering in natural, we thus prefer Cis over Des as a name for 1)
						#break
					except KeyError:
						pass
			return new

		# while there are any unnamed tones
		while len(set(xrange(self.scalesize)) - set(self.t2l.keys())) > 0:
			new = one_round(self.t2l, self.d2m)
			# if we can not generate anything new
			if len(new) == 0:
				warnings.warn("With given tone modificators, it is not possible to name all the tones!!")
				break
			self.t2l.update(new)

	def _init_parsing(self):
		from pyparsing import CaselessLiteral, OneOrMore, Or, ZeroOrMore, Empty

		def get_parsers_pairs(pa_fc, projdict):
			return [ (CaselessLiteral(label) if len(label) > 0 else Empty()).setParseAction( pa_fc(val) )
												for label, val  in projdict.iteritems() ]

		def num_tok(num):
			def pa(st, locn, toks):
				return num
			return pa

		def pa_sum(st, locn, toks):
			return sum(toks) % self.scalesize

		self.p_modi = OneOrMore(Or(get_parsers_pairs(num_tok, self.m2d)))
		self.p_tone = Or(imap( lambda x: (x + ZeroOrMore(self.p_modi)).setParseAction(pa_sum), get_parsers_pairs(num_tok, self.l2t)))
		self.p_chord_modi = Or(get_parsers_pairs(num_tok, self.chsuff2mask))

		def pach(st, locn, toks):
			assert len(toks) == 2
			return self._shift_mask(toks[0], toks[1])
		self.p_chord = (self.p_tone + self.p_chord_modi).setParseAction(pach)

	def _map_mask(self, fc, mask):
		return tuple(sorted(imap(lambda x : fc(x) % self.scalesize, mask)))

	def _shift_mask(self, shift, mask):
		return self._map_mask(lambda x : x + shift, mask)

	def _mask_to_base_n_chmask(self, mask):
		for basetone in mask:
			chmask = self._shift_mask( - basetone, mask)
			if chmask in self.mask2chsuff:
				yield basetone, chmask

	def test(self):
		print "\nChecking tones:"
		for x in [ 'C', 'Cis', 'Des', 'Fisis', 'C' + 'is'*12 ]:
			print x, ":", self.tone_name_to_tone(x)

		print "\nChecking chords:"
		for ch in [ "C", "Cmaj7", "Fisissus6", 'Em7', 'Gsus6', "Gis5+" ]:
			m = self.chordname_to_mask(ch)
			ch2 = self.mask_to_chordname(m)
			m2 = self.chordname_to_mask(ch2)
			print " -> ".join(imap(lambda x: str(x), [ch,m,ch2,m2]))

		m = (0,4,8)
		print "\nChecking all names for a mask", m, ":"
		for bt,msk in self._mask_to_base_n_chmask(m):
			for mskname in self.mask2chsuff[msk]:
				for tone in self.t2l[bt]:
					print tone + mskname

def ChordEvaluator():
	EvalFcs=[]
	def evalfc(f):
		def g(ch):
			ret = f(ch)
			return ret if ret != None else 0
		EvalFcs.append(g)
		return g
	def count_none(ch):
		ch = sorted(ch.ch_sf)
		up, down = 0, 0
		while up < len(ch) and ch[up][1] == None:
			up += 1
		while len(ch)-1-down >= 0  and ch[len(ch)-1-down][1] == None:
			down += 1
		middle = 0
		i = up
		state = 'no none'
		while True:
			i += 1
			if i >= len(ch):
				return up, middle, down
			if state == 'no none' and ch[i][1] == None:
				state = 'middle none'
			if state == 'middle none' and ch[i][1] != None:
				state = 'no none'
				middle += 1

	##
	##    The more, the better
	##
	@evalfc
	def finger_dist(ch):
		fg2fr = ch.finger2fret
		cnt = 0
		for lower_finger in xrange(1,ch.num_fingers-1):
			fr1, fr2 = fg2fr.get(lower_finger, None), fg2fr.get(lower_finger+1, None)	
			if fr1 != None and fr2 != None:
				if fr1 > fr2:
					assert False
					cnt += 20
				if fr2 - fr1 >= 2:
					cnt += 4*(fr2-fr1)
		return - cnt
	@evalfc
	def none_positions(ch):
		up, middle, down = count_none(ch)
		return - 8 * up - 20 * middle - 4 * down
	@evalfc
	def consecutive_strings_same_tone(ch):
		def tone(string):
			fr = ch.string2fret.get(string, None)
			if fr == None:
				return None
			return ch.instrument.tone(string, fr)
		cnt = 0
		for lower_string in xrange(len(ch.instrument.strings)-1):
			lw, up = tone(lower_string), tone(lower_string + 1)
			if lw != None and up != None and lw == up:
				ch.samestrings.append((lower_string, lower_string+1, lw))
				cnt -= 4
		return cnt
	@evalfc
	def how_much_do_fingers_hold(ch):
		fg = ch.finger2strings
		return -sum( 8*(len(fg[f]) if len(fg[f]) >= 2 else 0 ) for f in xrange(1,ch.num_fingers) )
	@evalfc
	def close_to_zero_fret(ch):
		return - ch.minfret
	@evalfc
	def num_fingers_used(ch):
		return - sum( ( 1 if len(strings) > 0 else 0 for strings in ch.finger2strings.itervalues() ) )
	@evalfc
	def span(ch):
		# prefer chords with small width
		return - ch.span
	@evalfc
	def no_pinky(ch):
		# my pinky is not as fast as other fingers
		return sum( ( -finger if (finger > 2 and len(strings) > 0) else 0 for finger,strings in ch.finger2strings.iteritems()) ) )
	@evalfc
	def finger_diff(ch):
		# two neighboring fingers should not hold string svery far from each other
		diff = 0
		#diff_mapper
		fh = ch.finger2strings
		# TODO
		for f in xrange(1,ch.num_fingers-1):
			fr1, fr2 = len(fh.get(f, ()), len(fh.get(f+1, ()))
			if fr1 and fr2:
				m1, m2 = max(fh[f]), max(fh[f+1])
				df = abs(m1-m2)
				if df > 2:
					diff += df * 1 #max( (f-2,1) )

		return - 1.2 * diff

# 	TODO
#	at se macka jednim prstem spis vsechny struny nez jen par vysokych (vysoke e)

	def EVALUATOR(ch):
		return map(lambda x: x(ch), EvalFcs)

	return EVALUATOR



class Instrument:
	"""
		Class representing an instrument.

		Provides mapping from string & fret -> tone
	"""
	def __init__(self, strings, tone_system, num_frets):
		'''
		strings is a tuple, from the bottommost,
			e.g. for a guitar
			strings = ('E','H','G','D','A','E')
			e.g. for a bass guitar
			strings = ('G','D','A','E')
		'''
		self.ts = tone_system
		self.string_names = strings
		self.strings = tuple(imap(tone_system.tone_name_to_tone, strings))
		self.num_frets = num_frets

	def get_fretboard(self):
		return [ set((fr,(zerofret + fr) % self.ts.scalesize) for fr in xrange(self.num_frets + 1)) for zerofret in self.strings]

	def tone(self, string, fret=None):
		if fret == None:
			fret = 0
		return (self.strings[string] + fret) % self.ts.scalesize

class ChordGenerator:
	def __init__(self, instrument, music_system, max_chord_span=2, num_fingers=4):
		self.instrument = instrument
		self.music_system = music_system
		self.max_chord_span = max_chord_span
		self.num_fingers = num_fingers
		self.EVAL = ChordEvaluator()

	def search_what_to_press(self, mask):
		'''
			Generates all the possibilities of what to press (ch_sf) such that
			only tones in mask (and all of them) are present.
		'''
		def filter_fretboard_by_chordmask(fretboard, mask):
			return [ set(ifilter(lambda pair: pair[1] in mask, st)) for st in fretboard ]
		# filter out tones that are not in chord
		fretboard = filter_fretboard_by_chordmask(self.instrument.get_fretboard(), mask )

		def dfs(chords_res, rest_of_tones, chordsofar=(), chordspan=(self.instrument.num_frets, 0), string=0):
			'''
				@chord_res - stack for results
				@rest_of_tones - remaining tones that must be present in the chord
				@chordsofar - 
			'''
			if string >= len(self.instrument.strings):
				if len(rest_of_tones) == 0:
					chords_res.append(chordsofar)
				return
			for fret, tone in fretboard[string] | set(((None,None),)):
				minfret, maxfret = chordspan
				# we want such chords that do not span across more then 3 (or self.max_chord_span) frets
				# no one has hand that wide... :-)
				if fret > 0:
					# too long
					if fret > minfret and fret - minfret > self.max_chord_span: continue
					if fret < maxfret and maxfret - fret > self.max_chord_span: continue
					# update min & max
					if fret < minfret: minfret = fret
					if fret > maxfret: maxfret = fret
				responsible=set((tone,)) & rest_of_tones
				rest_of_tones -= responsible
				dfs(chords_res, rest_of_tones, chordsofar + ((string,fret),), (minfret, maxfret), string+1)
				rest_of_tones |= responsible

		# Stack for results
		chords_res = []
		# The chord must contain all the tones in mask
		rest_of_tones = set(mask)

		dfs(chords_res, rest_of_tones)

		return chords_res

	def find_grips(self, chord):
		'''
			Generates all the possible finger assignments to a ch_sf.
		'''
		none_strings = list(filter(lambda x: x[1] == None, chord))
		chord = filter(lambda x: x[1] != None, chord)
		# a function ordering the (string, fret) pairs "lexicographically"
		def sweep_cmp(x,y):
			str1, fr1 = x
			str2, fr2 = y
			if fr1 != fr2:
				return cmp(fr1,fr2)
			return - cmp(str1,str2)
		def myprint(depth, *arg):
			pass
			#print ' '.join(["|"*depth] + arg)
		def search_grips(result_stack, index, assignment, sweepline, free_fingers=self.num_fingers, depth=0):
			myprint( depth, "fingers:", free_fingers)
			if index >= len(sweepline):
				result_stack.append(tuple(sorted(assignment+none_strings)))
				myprint( depth, "save res", result_stack[-1], ", fallback")
				return
			if free_fingers <= 0:
				myprint( depth, "fallback")
				return
			this_finger = self.num_fingers - free_fingers + 1
			this_string, this_fret = sweepline[index]

			# if we can try bare (and it is meaningful)
			max_index_on_this_fret = index
			while (max_index_on_this_fret + 1) < len(sweepline) and sweepline[max_index_on_this_fret+1][1] == this_fret:
				max_index_on_this_fret += 1
			# if it has effect (we press more than one string) and we do not block any other "higher" string
			if max_index_on_this_fret > index and not any(imap(lambda p: (p[0] < this_string) and (p[1] < this_fret), sweepline)):
				assindex = len(assignment)
				for string, fret in sweepline[index : max_index_on_this_fret + 1]:
					assignment.append((string, this_finger))
				myprint( depth, "bare")
				search_grips(result_stack, max_index_on_this_fret + 1, assignment, sweepline, free_fingers - 1, depth+1)
				myprint( depth, "bare back")
				del assignment[assindex:]

			'''
# TODO - make a better distinction between this and bare
			# if we can try pressing more consecutive strings with a single finger
			max_consecutive_index = index
			while (max_consecutive_index + 1) < len(sweepline):
				next_string, next_fret = sweepline[max_consecutive_index+1]
				if next_fret != this_fret:
					break
				if next_string != this_string - 1:
					break
				max_consecutive_index += 1
			if max_consecutive_index > index:
				assindex = len(assignment)
				for string, fret in sweepline[index : max_consecutive_index + 1]:
					assignment.append((string, this_finger))
				myprint( depth, "consecutive")
				search_grips(result_stack, max_consecutive_index + 1, assignment, sweepline, free_fingers - 1, depth +1)
				myprint( depth, "consecutive back")
				del assignment[assindex:]
			'''

			# normal pressing
			assindex = len(assignment)
			assignment.append((this_string, this_finger))
			myprint( depth, "normal")
			search_grips(result_stack, index + 1, assignment, sweepline, free_fingers - 1, depth+1)
			myprint( depth, "normal back")
			del assignment[assindex:]

			# try to skip some fingers
			myprint( depth, "skip")
			search_grips(result_stack, index, assignment, sweepline, free_fingers - 1, depth+1)
			myprint( depth, "skipback")

		# the pairs (string, fret) ordered lexicographically
		sweepline = sorted(chord, cmp=sweep_cmp)
		# index of the current string in sweepline
		index = 0
		assignment = []
		# assing the strings that should sound at zero fret to "None" - meaning that no finger
		# should hold them
		while index < len(sweepline) and sweepline[index][1] == 0:
			assignment.append((sweepline[index][0], None))
			index += 1

		result_stack = []
		# search for all possible grips of other strings & frets
		search_grips(result_stack, index, assignment, sweepline)

		return result_stack

	def evaluator(self, ch_n_g):
		return self.EVAL(ch_n_g)

	def shrink_head(self, head, fit, fitness_limit):
		if len(head) == 0:
			return []
		limit =	fit(head[0]) * fitness_limit
		return list(takewhile(lambda x: fit(x) >= limit, head))

	def find_chords(self, mask, fitness_limit=2):
		what_to_press = self.search_what_to_press(mask)
		chl = [ Chord(ch, grip, self.num_fingers, self.instrument)
						for ch in what_to_press for grip in self.find_grips(ch) ]
		
		fitnessd = dict( (ch, self.evaluator(ch)) for ch in chl)
		fit = lambda  ch : fitnessd[ch]
		chl.sort(key=fit)

		return self.shrink_head(chl, fit, fitness_limit)

	def find_chords_clever(self, mask, fitness_limit=2):
		what_to_press = self.search_what_to_press(mask)
		if len(what_to_press) == 0:
			return []

		fit = lambda  ch : self.evaluator(ch)

		ch_best_chng = {}
		for ch in what_to_press:
			grips = self.find_grips(ch)
			if len(grips) > 0:
				ch_best_chng[ch] =  max( Chord(ch, grip,  self.num_fingers, self.instrument)
											for grip in grips, key=fit)

		# which what_to_press are subset of each other:
		# e.g. A=(2, None, None, 3, 2, 2) is a "subset" of B
		#      B=(2,    3,    3, 3, 2, 2) but A is NOT subset of C
		#      C=(4,    3,    3, 4, 2, 2)
		filter_nn = lambda ch : set((s,f) for s, f in ch if f != None)
		def chord_cmp(ch1, ch2):
			# returns 	1  if ch1 > ch2 ( ch2 is a (strict) subset of ch1 )
			s1_nn, s2_nn = filter_nn(ch1), filter_nn(ch2)
			#if s1_nn == s2_nn:
			#	print ch1, ch2
			#assert s1_nn != s2_nn
			return s1_nn > s2_nn
		'''
		filter_nn = lambda ch : set((s,f) for s, f in ch if f != None)
		def chord_cmp(ch1, ch2, cache={}):
			# returns 	1  if ch1 > ch2 ( ch2 is a (strict) subset of ch1 )
			#if s1_nn == s2_nn:
			#	print ch1, ch2
			#assert s1_nn != s2_nn
			s1 = cache.get(ch1, None)
			s2 = cache.get(ch2, None)
			if not s1:
				s1 = filter_nn(ch1)
				cache[ch1] = s1
			if not s2:
				s2 = filter_nn(ch2)
				cache[ch2] = s2
			return ( s1 > s2 )
		'''


		from topological_sort import poset_to_edges, iterable_to_edge_dict, iter_edges, reverse_edge_dict, topological_sort

		V = what_to_press
		E = iterable_to_edge_dict(poset_to_edges(V, chord_cmp))
		rev_E = reverse_edge_dict(E)
		topo_V = topological_sort((V,E))

		dead_chs = set()
		for ch in topo_V:
			my_best = ch_best_chng.get(ch,None)
			if my_best != None:
				my_parents = E.get(ch, ())
				for parent in my_parents:
					par_best = ch_best_chng.get(parent,None)
					if par_best == None or fit(par_best) < fit(my_best):
						# this means that `parent' is becoming son of all my parents
						ch_best_chng[parent] = my_best
					if par_best != None and fit(par_best) > fit(my_best):
						dead_chs.add(my_best)

		head = []
		for ch in V:
			if len(E.get(ch,())) == 0:
				chng = ch_best_chng.get(ch,None)
				if chng != None and not chng in dead_chs:
					head.append(chng)

		head = [k for k, g in groupby(sorted(head,reverse=True, key=fit))]

		return self.shrink_head(head, fit, fitness_limit)


def main():
	base_tones = [('C',0),('D',2),('E',4),('F',5),('G',7),('A',9),('B',10),('H',11)]
	base_tones_en = [('C',0),('D',2),('E',4),('F',5),('G',7),('A',9),('B',11)]
	tone_modificators = [('#',1), ('b',-1), ('is',1), ('es', -1)]
	chord_masks = [(['', 'major'],(0,4,7)),(['mi','m'],(0,3,7)),(['5+','+','aug'],(0,4,8)),('sus4',(0,5,7)),(['6','sus6','add6'],(0,4,7,9)),
					(['maj','maj7'],(0,4,7,11)),(['7'],(0,4,7,10)),(['m7', 'mi7'],(0,3,7,10)),(['dim','dim7'],(0,3,6,9)),
					(['5b','5-'],(0,4,6)),(['mi6', 'm6','min6'],(0,3,7,9)), (['9'],(0,4,7,10,2))]

	t = MusicSystem(12, ManyToOneNamer(base_tones_en), ManyToOneNamer(tone_modificators), ManyToOneNamer(chord_masks))

	t.test()

	return

	frequent_tones = ['C','D','E','F','G','A']
	frequent_chords = frequent_tones + [ tone + "mi" for tone in frequent_tones ] + ["D7","Amaj","Dmaj","A7","E7","Em7"]

	#	Guitar
	#i = Instrument(('E','H','G','D','A','E'),t,8)
	i = Instrument(('E','B','G','D','A','E'),t,8)
	c = ChordGenerator(i)

	#	Bass guitar
	#i = Instrument(('G','D','A','E'),t,8)
	#c = ChordGenerator(i, max_chord_span=1)

	def f():
		chngs = [ c.find_chords_clever(t.chordname_to_mask(chord),2) for chord in frequent_chords ]
		return chngs
	#f()

	def find_chord(chords):
		for chord in chords:
			chngs = c.find_chords_clever(t.chordname_to_mask(chord),3)
			print chord
			print
			if len(chngs):
				for x in  chngs:
					print x
					print
			else:
				print "NONE!!!!!!!!!!!!!"
			print "---------------------------"

	import sys

	find_chord(sys.argv[1:])

if __name__ == '__main__':
	main()
