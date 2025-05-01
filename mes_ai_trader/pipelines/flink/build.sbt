name := "mes-trader-flink-jobs"
version := "0.1.0"
scalaVersion := "2.12.15"
organization := "com.mesaitrader"

val flinkVersion = "1.16.1"
val awsKinesisVersion = "2.2.0"

resolvers ++= Seq(
  "Apache Development Snapshot Repository" at "https://repository.apache.org/content/repositories/snapshots/",
  Resolver.mavenLocal
)

/* Dependencies */
libraryDependencies ++= Seq(
  // Core Flink dependencies
  "org.apache.flink" %% "flink-scala" % flinkVersion % "provided",
  "org.apache.flink" %% "flink-streaming-scala" % flinkVersion % "provided",
  
  // Kinesis connector
  "org.apache.flink" %% "flink-connector-kinesis" % flinkVersion,
  "software.amazon.kinesis" % "amazon-kinesis-client" % awsKinesisVersion,
  
  // JSON processing
  "org.apache.flink" %% "flink-json" % flinkVersion,
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.13.4",
  
  // Metrics and monitoring
  "org.apache.flink" %% "flink-metrics-core" % flinkVersion % "provided",
  
  // Logging
  "org.slf4j" % "slf4j-api" % "1.7.36",
  "org.slf4j" % "slf4j-log4j12" % "1.7.36" % "runtime",
  
  // Testing
  "org.scalatest" %% "scalatest" % "3.2.14" % "test",
  "org.apache.flink" %% "flink-test-utils" % flinkVersion % "test"
)

// Assembly settings for creating a fat JAR
assembly / assemblyJarName := s"${name.value}-${version.value}.jar"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case "log4j.properties" => MergeStrategy.first
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}

// Scala compiler options
scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-language:implicitConversions",
  "-language:postfixOps"
)

// Exclude Scala library from assembly
assembly / assemblyOption := (assembly / assemblyOption).value.copy(
  includeScala = false
)

// Set the main class
Compile / mainClass := Some("com.mesaitrader.flink.OHLCVJob") 