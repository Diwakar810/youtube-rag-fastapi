from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
parser=StrOutputParser()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.1)

def load_transcript():
    test_transcript = """
Jay Shetty: Welcome back everyone to On Purpose. Today we have an incredible guest, Vinh Giang, who's known as one of the top communication experts in the world. Vinh, welcome to the show.

Vinh Giang: Jay, thank you so much for having me. It's an absolute honor.

Jay Shetty: Now Vinh, I want to dive right in because you work with some of the biggest names - Fortune 500 companies, you've done TEDx talks that have millions of views. But what I find most fascinating is that you actually struggled with social anxiety for years. Can you take us back to that time?

Vinh Giang: Absolutely. So picture this - I'm 19 years old, I'm at university, and I'm terrified of public speaking. Not just nervous, but physically ill. My hands would shake, my voice would crack, I'd sweat through my shirt. I remember one presentation where I literally ran off stage mid-sentence because I couldn't handle the eye contact from 30 classmates.

Jay Shetty: Wow. And yet here you are today, teaching thousands of people how to communicate confidently. What was the turning point?

Vinh Giang: It was actually watching magic shows. Sounds weird right? But I noticed something fascinating - magicians don't get nervous on stage. They walk out, they command attention, they pause dramatically, and the audience hangs on every word. So I started studying them obsessively.

Jay Shetty: That's brilliant. So what did you learn from the magicians that changed everything?

Vinh Giang: The first big lesson was about pauses. Most people who are anxious fill every silence with ums, ahs, and nervous chatter. But magicians weaponize silence. They pause after big moments, and that pause creates tension, curiosity, anticipation. 

Let me demonstrate. [pause] Did you feel that? Your brain just went 'What's he going to say next?' That's the power of pause. Anxious people fear silence, confident people embrace it.

Jay Shetty: I felt that completely. So practically speaking, if someone's listening right now and they get anxious in conversations, how do they start using pauses?

Vinh Giang: Start small. Next time you're talking to someone and you make an important point, just stop talking. Count to two in your head. Don't fill the silence. Watch what happens - the other person leans in, they nod, they process what you said. You've just become 10x more charismatic in 2 seconds.

Jay Shetty: That's so powerful. What else did the magicians teach you?

Vinh Giang: The second lesson was about breathing. Here's something most people don't realize - when you're anxious, your breathing becomes shallow and fast, which triggers your fight-or-flight response. Magicians use what's called diaphragmatic breathing. 

Try this with me right now. Put your hand on your stomach. Now breathe in through your nose for 4 seconds so your stomach expands, hold for 4, exhale through your mouth for 6. Feel that? Your nervous system just calmed down.

Jay Shetty: I'm doing it right now and I can literally feel my shoulders dropping. 

Vinh Giang: Exactly. And here's the communication hack - when you need to make a strong point, take that deep belly breath first, then speak. Your voice drops an octave, you sound more authoritative, and most importantly, you feel more in control.

Jay Shetty: This is gold. Now let's talk about the physical side of anxiety. A lot of our listeners get red-faced, sweaty palms, shaky voice. What do you tell your clients about this?

Vinh Giang: First, understand this - your body doesn't know the difference between real danger and social anxiety. Evolution wired us to blush, shake, sweat when facing predators. Your brain thinks 'Oh no, 20 people are looking at me, must be a lion!' 

The trick is to reframe it. Instead of fighting the physical symptoms, lean into them. I tell my clients: 'When you feel the shake starting, smile and say "Isn't it funny how excited I get about this topic?"' 

What happens? You just named the elephant in the room. Everyone laughs, tension breaks, and suddenly you're the confident one.

Jay Shetty: Genius. So you're saying don't hide the anxiety, own it?

Vinh Giang: 100%. Here's another example. I was doing a keynote for 2000 CEOs. Five minutes in, my voice starts cracking. Old Vinh would have panicked. New Vinh pauses, chuckles, and says 'You know, even after 10 years of this, I still get butterflies before speaking to this many leaders. Isn't that wild?' 

The room erupted in applause. They connected with my vulnerability. Suddenly I wasn't the 'expert' anymore, I was human. That's when the magic happens.

Jay Shetty: That's such a powerful reframe. Vinh, let's talk about eye contact because this terrifies so many people. They either stare too intensely or look everywhere but at the person.

Vinh Giang: Perfect topic. Here's the three-second rule. Most anxious people either do the 'laser stare' which feels aggressive, or the 'avoider' who looks at noses, foreheads, walls. 

The fix: Look someone directly in the eye for exactly 3 seconds, then look away briefly to the right, then back. Why three seconds? It's long enough to connect, short enough to feel natural. Practice this at coffee shops. You'll notice waitresses, baristas suddenly treat you differently.

Jay Shetty: Why to the right specifically?

Vinh Giang: Psychology. Looking right accesses the visual memory part of your brain, so it looks like you're recalling something thoughtful rather than staring. Looking left accesses visual construction - your brain thinks you're making stuff up.

Jay Shetty: Okay, our listeners are taking furious notes right now. What about voice? A lot of people hate their nervous voice.

Vinh Giang: Voice is 38% of communication impact. Three things: pace, volume, pitch. Anxious people speak too fast, too quiet, pitch too high.

Fix 1: Slow your pace by 20%. Practice reading aloud with a metronome at 120 beats per minute.
Fix 2: Before important sentences, drop your volume slightly. People lean in.
Fix 3: End sentences with a downward inflection. Upward sounds like questions, downward sounds like authority.

Jay Shetty: Vinh, this is incredibly practical. Let's do a quick exercise for everyone listening. Imagine you're at a networking event, you need to introduce yourself confidently. Walk us through it.

Vinh Giang: Perfect. Three steps:

1. Belly breath - hand on stomach, inhale 4, hold 4, exhale 6
2. Three-second eye contact with your target
3. Smile and speak slowly: 'Hi [name], I've been excited to meet you. I saw your work on [specific thing].'

Pause. Watch their face light up. You've just made them feel seen and important.

Jay Shetty: Beautiful. Now Vinh, I know you also teach about body language. What's the number one mistake people make?

Vinh Giang: Hands in pockets. Or crossed arms. Both signal 'closed off, don't approach me.' 

The power pose: Stand with feet shoulder-width, hands visible either gesturing or on hips. Chin slightly up. This isn't just for confidence - research shows it actually changes your hormone levels. Higher testosterone, lower cortisol within 2 minutes.

Jay Shetty: Amy Cuddy's power posing research!

Vinh Giang: Exactly. But here's the communication twist - open body language doesn't just make you confident, it makes other people trust you instantly. 

Try this experiment: Next time you're in a meeting, notice who has open posture vs closed. Who gets interrupted? Who leads the conversation?

Jay Shetty: Brilliant observation. Vinh, let's address the inner game. A lot of people intellectually get this, but emotionally they still feel 'I'm not good enough.'

Vinh Giang: This is the core issue. Here's what I learned from studying top performers - they all had the same secret: They stopped trying to be confident, and started focusing on being generous.

Instead of 'I hope I don't mess up,' they think 'How can I serve this audience?' When your focus shifts from self-protection to service, anxiety disappears.

Jay Shetty: That's profound. Can you give us a practical way to make that mindset shift?

Vinh Giang: Yes. Before every talk, I write down three specific ways I can add value to that audience. 'I want to give them one tool they can use tomorrow.' 'I want them to leave feeling hopeful.' 'I want to make them laugh twice.'

When you have a clear service mission, your brain stops obsessing about your performance.

Jay Shetty: Vinh, this has been incredible. Before we wrap up, what's one exercise our listeners can do tonight that will immediately improve their communication?

Vinh Giang: The mirror pause challenge. Stand in front of a mirror. Say one sentence about something you're passionate about. Then pause and hold eye contact with yourself for 5 seconds. No looking away. 

Most people can't do it. They look down, they fidget. Master this, and you'll master any room.

Jay Shetty: Powerful. Vinh Giang, thank you so much for sharing your wisdom with us. Where can people find you?

Vinh Giang: My website vinhgiang.com or just search my TEDx talk 'The science of magic.' 

Jay Shetty: Incredible. Everyone, if you're feeling anxious about communication, start with one thing from today. The pause, the breathing, the eye contact. Small changes, massive impact.

Thank you for being here. We'll see you in the next episode.
"""
    return test_transcript


def load_transcript1(url:str):
    video_id=url
    try:
        transcript_list=YouTubeTranscriptApi().fetch(video_id)
        transcript= " ".join(chunk.text  for chunk in transcript_list.snippets)
        return transcript
    
    except TranscriptsDisabled:
        print("Transcript is disabled for this video")
    except Exception as e:  
        print(f"There is an Exception: {e}")  # Fixed: Use f-string
    return None


def ask_question(url:str, question:str):

    transcript=load_transcript()
    docs = [Document(page_content=transcript)]
    splitted_transcript = splitter.split_documents(docs)
    # splitted_transcript=splitter.split_text(transcript)
    vector_store=FAISS.from_documents(documents=splitted_transcript,embedding=embedding)
    retriever=vector_store.as_retriever(search_kwargs={"k":5})
    context=retriever.invoke(question)
    prompt=PromptTemplate(
        template="""
        You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.
                {context}
                Question: {question}
        """,
        input_variables=["context","question"]
    )
    chain=prompt |llm | parser
    result=chain.invoke({"context":context,"question":question})

    return result


print(ask_question("ru44DngJYoA"," what is this video about"))