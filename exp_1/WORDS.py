class WORDSSet:
    # def __init__(self):
    # 定义指令动词
    instruct1 = [
        "translate", "explain", "summarize", "retrieve",
        "revise", 'generate', 'describe', 'classify', 'create',
        "evaluate", "correct", "develop",
        "identify", "analyze", "compose", "demonstrate", "interpret",
        "design", "solve", "follow", "clarify", "say", "help", "act",
        "recommend", "estimate", "edit", "format", "repeat"
    ]
    instruct2 = [
        "write", "give", "find", "create", "make", "describe", "design",
        "generate", "classify", "have", "explain", "tell", "identify",
        "output", "predict", "detect"
    ]
    instruct3 = ["give", "make", "solve", "create", "look", "write", "see", "add", "return",
                 "take", "set", "think", "calculate", "say", "call", "let", "run", "read",
                 "understand", "define", "follow", "start", "like", "come", "ask", "change",
                 "contain", "include", "consider", "base", "generate", "help", "require",
                 "expect", "implement", "check", "pass", "put", "provide", "assume", "keep",
                 "explain", "apply", "remove", "learn", "compute", "build", "tell", "throw",
                 "test", "leave", "save", "extend", "answer", "please", "convert", "fix",
                 "turn", "determine", "figure", "suppose", "move", "handle", "describe",
                 "produce", "become", "play", "wonder", "compare", "appear", "avoid",
                 "delete", "prove"]

    instruction_verbs = set(instruct1 + instruct2 + instruct3)

    # 礼貌标记词列表
    politeness_markers = {
        # 请求词
        'please', 'kindly', 'may',

        # 感谢词
        'thanks', 'thank', 'cheers', 'appreciate', 'grateful',

        # 道歉词
        'sorry', 'apologies', 'pardon', 'excuse', 'forgive',

        # 问候词
        'hi', 'hello', 'greetings', 'welcome', 'regards',

        # 尊称词
        'sir', 'madam', 'ma\'am', 'miss', 'mr', 'mrs', 'ms',

        # 祝福/祝贺词
        'congratulations', 'congrats', 'bless', 'wishes',

        # 副词类礼貌词
        'respectfully', 'sincerely', 'cordially', 'humbly',
        'graciously', 'gently', 'warmly',

        # 口语化礼貌词
        'dear', 'mate', 'friend', 'folks',

        # 短语形式（保留原有）
        'thank you', 'excuse me', 'pardon me', 'would you', 'could you',
        'may i', 'might i', 'might you', 'should you',
        'if you don\'t mind', 'if possible', 'when you get a chance',
        'at your convenience', 'if you have time', 'when convenient',
        'would it be possible', 'if you could',
        'i would appreciate', 'i\'d appreciate', 'much appreciated',
        'many thanks', 'appreciate it'
    }

    # 结构化标记词列表
    structure_markers = {
        # 顺序词
        'first', 'firstly', 'second', 'secondly', 'third', 'thirdly',
        'fourth', 'fifth', 'next', 'then', 'lastly', 'finally',
        'afterwards', 'subsequently', 'eventually',
        'step 1', 'step 2', 'step 3', 'step 4',

        # 添加/递进类
        'also', 'besides', 'furthermore', 'moreover', 'additionally',
        'in addition', 'as well', 'likewise', 'similarly', 'equally',
        'what\'s more', 'not only', 'along with',

        # 对比/转折类
        'however', 'but', 'yet', 'nevertheless', 'nonetheless',
        'although', 'though', 'even though', 'despite', 'in spite of',
        'on the other hand', 'in contrast', 'conversely', 'whereas',
        'while', 'instead', 'rather', 'alternatively',

        # 因果类
        'therefore', 'thus', 'hence', 'consequently', 'accordingly',
        'as a result', 'for this reason', 'because', 'since', 'so',
        'due to', 'owing to', 'thanks to', 'as', 'for',

        # 举例类
        'for example', 'for instance', 'such as', 'like',
        'namely', 'specifically', 'in particular', 'especially',
        'to illustrate', 'as an illustration',

        # 强调类
        'indeed', 'in fact', 'actually', 'certainly', 'obviously',
        'clearly', 'undoubtedly', 'without doubt', 'of course',
        'above all', 'most importantly', 'primarily',

        # 总结类
        'in conclusion', 'to conclude', 'to summarize', 'in summary',
        'overall', 'in short', 'in brief', 'to sum up',
        'all in all', 'on the whole', 'in general',

        # 列表引导词
        'here are', 'here\'s', 'let me', 'i\'ll', 'the following',
        'below', 'above', 'as follows', 'listed below',

        # 时间关系词
        'meanwhile', 'simultaneously', 'at the same time',
        'before', 'after', 'during', 'until', 'when', 'whenever'
    }

    # 疑问词列表
    question_words = {
        'what', 'how', 'why', 'where', 'when', 'who', 'which', 'whose', 'whom',
        'whatever', 'however', 'wherever', 'whenever', 'whoever', 'whichever',
        'whether', 'whence', 'whither', 'whereby', 'wherein', 'whereof', 'wherefor',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
        'do', 'does', 'did', 'is', 'are', 'was', 'were', 'am', 'been', 'being',
        'have', 'has', 'had', 'having'
    }

    # 实验3：确定性/不确定性词汇
    certainty_markers = {
        'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly', 'absolutely',
        'surely', 'indeed', 'exactly', 'precisely', 'unquestionably', 'indubitably',
        'always', 'never', 'must', 'will', 'shall', 'fact', 'true', 'correct', 'proven',
        'invariably', 'unfailingly', 'positively', 'decidedly', 'unequivocally',
        'categorically', 'conclusively', 'indisputably', 'irrefutably', 'incontrovertibly',
        'assuredly', 'doubtlessly', 'evidently', 'manifestly', 'patently', 'plainly',
        'demonstrably', 'veritably', 'genuinely', 'authentically', 'actually', 'really',
        'truly', 'factually', 'objectively', 'empirically', 'verifiably', 'confirmedly',
        'established', 'confirmed', 'validated', 'verified', 'substantiated', 'documented',
        'certain', 'sure', 'definite', 'clear', 'obvious', 'evident', 'apparent',
        'unmistakable', 'unambiguous', 'explicit', 'distinct', 'guaranteed', 'assured'
    }

    uncertainty_markers = {
        'might', 'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'unlikely',
        'may', 'could', 'would', 'should', 'seem', 'seems', 'seemed', 'appears',
        'appeared', 'suggests', 'suggested', 'indicates', 'indicated', 'potentially',
        'presumably', 'supposedly', 'allegedly', 'reportedly', 'apparently', 'ostensibly',
        'conceivably', 'plausibly', 'feasibly', 'tentatively', 'provisionally',
        'hypothetically', 'theoretically', 'speculatively', 'arguably', 'debatably',
        'questionably', 'doubtfully', 'uncertainly', 'ambiguously', 'vaguely',
        'roughly', 'approximately', 'nearly', 'almost', 'somewhat', 'rather',
        'fairly', 'quite', 'pretty', 'sort', 'kind', 'relatively', 'comparatively',
        'suppose', 'assume', 'guess', 'think', 'believe', 'feel', 'suspect',
        'imagine', 'reckon', 'expect', 'anticipate', 'estimate', 'approximate',
        'possible', 'probable', 'potential', 'contingent', 'conditional', 'dependent',
        'uncertain', 'unclear', 'unsure', 'indefinite', 'indeterminate', 'ambiguous',
        'tentative', 'provisional', 'temporary', 'qualified', 'partial', 'incomplete'
    }

    # 语义转折词
    transition_markers = {
        'however', 'but', 'yet', 'nevertheless', 'nonetheless', 'although', 'though',
        'despite', 'while', 'whereas', 'conversely', 'instead', 'rather', 'alternatively',
        'still', 'even', 'notwithstanding', 'albeit', 'except', 'besides', 'moreover',
        'furthermore', 'additionally', 'also', 'too', 'likewise', 'similarly', 'equally',
        'correspondingly', 'meanwhile', 'simultaneously', 'concurrently', 'subsequently',
        'thereafter', 'beforehand', 'previously', 'formerly', 'lately', 'recently',
        'presently', 'currently', 'now', 'then', 'next', 'finally', 'ultimately',
        'eventually', 'consequently', 'therefore', 'thus', 'hence', 'accordingly',
        'otherwise', 'else', 'contrarily', 'oppositely', 'inversely', 'reciprocally',
        'paradoxically', 'ironically', 'surprisingly', 'unexpectedly', 'remarkably',
        'notably', 'significantly', 'importantly', 'essentially', 'basically',
        'fundamentally', 'primarily', 'chiefly', 'mainly', 'mostly', 'generally',
        'typically', 'usually', 'normally', 'commonly', 'frequently', 'often',
        'sometimes', 'occasionally', 'rarely', 'seldom', 'hardly', 'scarcely'
    }

    # 实验4：完整答案相关标记
    conclusion_markers = {
        'conclusion', 'conclude', 'summary', 'summarize', 'summarizing', 'summarized',
        'overall', 'finally', 'therefore', 'thus', 'hence', 'consequently',
        'accordingly', 'ultimately', 'eventually', 'lastly', 'conclusively',
        'altogether', 'collectively', 'cumulatively', 'totally', 'entirely',
        'completely', 'fully', 'wholly', 'absolutely', 'utterly', 'thoroughly',
        'comprehensively', 'exhaustively', 'definitively', 'decisively', 'terminally',
        'ending', 'closing', 'finishing', 'completing', 'culminating', 'terminating',
        'ceasing', 'stopping', 'halting', 'concluding', 'finalizing', 'wrapping',
        'recapping', 'reviewing', 'reiterating', 'restating', 'rephrasing',
        'brief', 'short', 'concise', 'succinct', 'compact', 'condensed',
        'end', 'finish', 'complete', 'close', 'terminate', 'cease', 'stop'
    }

    # 任务完成标记词
    task_completion_markers = {
        'steps', 'process', 'method', 'approach', 'solution', 'answer', 'result',
        'outcome', 'complete', 'completed', 'completing', 'finished', 'finishing',
        'done', 'achieved', 'achieving', 'accomplished', 'accomplishing', 'resolved',
        'resolving', 'solved', 'solving', 'addressed', 'addressing', 'handled',
        'handling', 'managed', 'managing', 'executed', 'executing', 'performed',
        'performing', 'implemented', 'implementing', 'conducted', 'conducting',
        'carried', 'carrying', 'fulfilled', 'fulfilling', 'satisfied', 'satisfying',
        'met', 'meeting', 'attained', 'attaining', 'reached', 'reaching', 'obtained',
        'obtaining', 'secured', 'securing', 'gained', 'gaining', 'acquired',
        'acquiring', 'delivered', 'delivering', 'provided', 'providing', 'produced',
        'producing', 'generated', 'generating', 'created', 'creating', 'developed',
        'developing', 'established', 'establishing', 'built', 'building', 'constructed',
        'constructing', 'formed', 'forming', 'made', 'making', 'prepared', 'preparing',
        'task', 'goal', 'objective', 'target', 'aim', 'purpose', 'mission',
        'achievement', 'accomplishment', 'completion', 'fulfillment', 'realization',
        'success', 'successful', 'successfully', 'effectively', 'efficiently'
    }

    # 情态动词
    modal_words = {
        # Core modal verbs
        "can", "could", "would", "should", "may", "might", "must",
        "will", "shall", "ought", "need", "dare",

        # Semi-modals and modal expressions (single words)
        "cannot", "won't", "wouldn't", "shouldn't", "couldn't", "can't",
        "mustn't", "needn't", "daren't", "shan't", "mightn't", "mayn't",
    }