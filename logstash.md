1. root파일 안 logs 디렉토리 생성
2. would_you_loan/src/main/resources/logback-spring.xml
```
#prometaus
management.endpoints.web.exposure.include=beans,env,health,info,metrics,mappings,prometheus
management.endpoint.health.show-details=always
management.health.probes.enabled=true
```
3. would_you_loan/pom.xml
```
<!-- Logstash-Logback Encoder -->
<dependency>
 <groupId>net.logstash.logback</groupId>
 <artifactId>logstash-logback-encoder</artifactId>
 <version>7.4</version>
</dependency>

<dependency>
 <groupId>io.micrometer</groupId>
 <artifactId>micrometer-registry-prometheus</artifactId>
</dependency>
```

4. would_you_loan/src/main/resources/logback-spring.xml
```
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property resource="application.properties"/>

    <!-- 로그 출력 형식 정의 -->
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder class="ch.qos.logback.classic.encoder.PatternLayoutEncoder">
            <pattern>%msg, %d{yyyy-MM-dd HH:mm:ss}%n</pattern>
        </encoder>
    </appender>

    <property name="LOG_PATH" value="./logs"/>
    <property name="LOG_FILE_NAME" value="api"/>
    <property name="ERR_LOG_FILE_NAME" value="api-error"/>

    <!-- API 로깅 -->
    <appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
        <filter class="ch.qos.logback.classic.filter.LevelFilter">
            <level>INFO</level>
            <onMatch>ACCEPT</onMatch>
            <onMismatch>DENY</onMismatch>
        </filter>
        <file>${LOG_PATH}/${LOG_FILE_NAME}.log</file>
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{36} - %msg%n
            </pattern>
        </encoder>

        <!-- 하루에 한번 압축 후 보관, 최대 30일, 1GB까지 보관 -->
        <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
            <fileNamePattern>${LOG_PATH}/${LOG_FILE_NAME}.%d{yyyy-MM-dd}.gz</fileNamePattern>
            <maxHistory>30</maxHistory>
            <totalSizeCap>1GB</totalSizeCap>
        </rollingPolicy>
    </appender>

    <appender name="LOGSTASH" class="net.logstash.logback.appender.LogstashTcpSocketAppender">
        <destination>regularmark.iptime.org:45001</destination>
        <encoder charset="UTF-8" class="net.logstash.logback.encoder.LogstashEncoder" />
    </appender>

    <!-- 로그 레벨 정의 -->
    <root level="INFO">
        <appender-ref ref="CONSOLE"/>
        <appender-ref ref="FILE"/>
        <appender-ref ref="LOGSTASH"/>
    </root>

</configuration>
```
