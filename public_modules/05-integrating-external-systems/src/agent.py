import logging
import aiohttp

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
    mcp,
    llm,
    stt,
    tts
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://shayne.app/weather?location={location}") as response:
                    if response.status == 200:
                        data = await response.json()
                        condition = data.get("condition", "unknown")
                        temperature = data.get("temperature", "unknown")
                        unit = data.get("unit", "degrees")
                        return f"{condition} with a temperature of {temperature} {unit}"
                    else:
                        logger.error(f"Weather API returned status {response.status}")
                        return "Weather information is currently unavailable for this location."
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return "Weather service is temporarily unavailable."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        llm=llm.FallbackAdapter(
            [
                openai.LLM(model="gpt-4o-mini"),
                openai.LLM(model="gpt-4.1"),
            ]
        ),
        stt=stt.FallbackAdapter(
            [
                deepgram.STT(filler_words=True),
                openai.STT(),
            ]
        ),
        tts=tts.FallbackAdapter(
            [
                deepgram.TTS(),
                openai.TTS(),
            ]
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        mcp_servers=[
            mcp.MCPServerHTTP(url="http://shayne.app/sse"),
        ],
    )

    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
