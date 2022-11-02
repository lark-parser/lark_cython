from lark import Lark, Tree
import lark_cython


def test_minimal():
	parser = Lark('!start: "a" "b"', parser='lalr', _plugins=lark_cython.plugins)
	res = parser.parse("ab")
	assert isinstance(res, Tree)
	assert all(isinstance(t, lark_cython.Token) for t in res.children)
	assert [t.value for t in res.children] == ['a', 'b']

def test_lark_meta_propagation():
	parser = Lark("""
	    start: INT*

	    COMMENT: /#.*/

	    %import common (INT, WS)
	    %ignore COMMENT
	    %ignore WS
	""", parser="lalr", _plugins=lark_cython.plugins, propagate_positions=True)
	res = parser.parse("""
		1 2 3  # hello
		# world
		4 5 6
		""")
	tokens = res.children
	assert isinstance(res, Tree)
	assert res.meta.line == 2
	assert all(isinstance(t, lark_cython.Token) for t in tokens)
	assert all(hasattr(t, "__lark_meta__") for t in tokens)
	assert all(t.__lark_meta__() == t for t in tokens)
	assert tokens[0].line == tokens[1].line == tokens[2].line == 2
	assert tokens[0].end_line == tokens[1].end_line == tokens[2].end_line == 2
	assert tokens[3].line == tokens[4].line == tokens[5].line == 4
	assert tokens[3].end_line == tokens[4].end_line == tokens[5].end_line == 4
	assert tokens[0].column == tokens[3].column == 3
	assert tokens[1].column == tokens[4].column == 5
	assert tokens[2].column == tokens[5].column == 7

def test_no_placeholders():
	parser = Lark('!start: "a" ["b"]', parser='lalr', _plugins=lark_cython.plugins, maybe_placeholders=True)

	assert len(parser.parse("a").children) == 2
	assert len(parser.parse("ab").children) == 2

	parser = Lark('!start: "a" ["b"]', parser='lalr', _plugins=lark_cython.plugins, maybe_placeholders=False)

	assert len(parser.parse("a").children) == 1
	assert len(parser.parse("ab").children) == 2

def test_start():
	parser = Lark('!x: "a" "b"', parser='lalr', _plugins=lark_cython.plugins, start='x')
	res = parser.parse("ab")
	assert [t.value for t in res.children] == ['a', 'b']

def test_lexer_callbacks():
	comments = []

	parser = Lark("""
	    start: INT*

	    COMMENT: /#.*/

	    %import common (INT, WS)
	    %ignore COMMENT
	    %ignore WS
	""", parser="lalr", _plugins=lark_cython.plugins, lexer_callbacks={'COMMENT': comments.append})


	res = parser.parse("""
		1 2 3  # hello
		# world
		4 5 6
		""")

	assert isinstance(res.children[0], lark_cython.Token)
	assert isinstance(comments[0], lark_cython.Token)
	assert len(comments) == 2